"""AM-I Support dynamic parameter identification with soromox.

Python/JAX port of identification.m: loads the exported experiment
(2026-06-08-11-59-18_py.mat), scales/permutes the recorded chamber
pressures, builds the ISupport model with the nominal (initial-guess)
parameters, and identifies [E, Eta, Rho] (Poisson frozen, pi_mask
[1,1,1,0]) by matching the simulated tip trajectory to the Vicon data.

Differences w.r.t. MATLAB, on purpose:
* soromox's ISupport takes chamber PRESSURES [Pa] directly (chamber
  geometry lives in ISupportParams), so the actuation scaling is
  bar->Pa + air-leak gains + permutation, WITHOUT the chamber-surface
  factor used for SoRoSim (which expects forces).
* gradients through the rollout come from autodiff. NOTE: this soromox
  build differentiates its rollout with a custom_vjp, so FORWARD-mode
  autodiff (jacfwd / optimistix 'lm') is unavailable; REVERSE mode
  (grad / jacrev / optax 'adam') works, but the diffrax adjoint needs a
  FINITE max_steps to size its memory. Hence max_steps is set below and
  the autodiff path is 'adam'.
"""

import os
# execute this command here: export XLA_PYTHON_CLIENT_PREALLOCATE=false
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import scipy.io as sio
from diffrax import PIDController, Tsit5

jax.config.update("jax_enable_x64", True)

from soromox.systems import ISupport, ISupportParams, ISupportStructure

from gvs_identification import (
    DynamicGVSIdentification,
    generate_time_window_weights,
    _central_fd_jacobian,          # module-level FD helper, for the AD cross-check
)

import equinox as eqx
from soromox.systems.pcs.isupport import _expand_isupport_layout

RESULTS_DIR = "amisupport_dataset/results_identification"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Reverse-mode autodiff through the diffrax rollout needs a FINITE bound on
# the number of solver steps (RecursiveCheckpointAdjoint cannot size its
# memory when max_steps=None). This must cover the steps the adaptive solver
# takes across the whole window; raise it if you hit a max_steps error.
ROLLOUT_MAX_STEPS = 16 ** 4      # = 65536

# =====================================================================
# Experimental data (same export used by the simulation script)
# =====================================================================
data = sio.loadmat("amisupport_dataset/2026-06-08-11-59-18_py.mat",
                   squeeze_me=True)
poses_data = np.asarray(data["poses_data"])          # (7, N, M)
merged_time = np.asarray(data["merged_time"]).squeeze()
act_values = np.asarray(data["actuation_input"], dtype=np.float64)  # (6, M)
N = int(data["N"])

# Zero the clock like exp_time in identification.m
act_time = merged_time - merged_time[0]

# =====================================================================
# Identification window: 15.22 s -> 60.22 s (identification.m)
# =====================================================================
start_time, end_time = 15.22, 16.22
start_idx = int(np.argmin(np.abs(merged_time - start_time)))
end_idx = int(np.argmin(np.abs(merged_time - end_time)))

exp_time = merged_time[start_idx:end_idx + 1] - merged_time[start_idx]
poses_window = poses_data[:, :, start_idx:end_idx + 1]

# Optional decimation to speed up the first runs (residuals on a coarser
# grid). stride = 1 reproduces the MATLAB setup exactly.
stride = 1
exp_time = exp_time[::stride]
poses_window = poses_window[:, :, ::stride]

# =====================================================================
# Actuation scaling: bar -> Pa, air-leak gains, channel permutation.
# (identification.m also multiplies by the chamber surface because
# SoRoSim wants forces; soromox wants pressures, so we do not.)
# =====================================================================
bar2pa = 1.0e5
p_real_leaks = np.array([2.33, 2.35, 2.26, 2.09, 2.63, 2.32])
air_leaks = np.diag(p_real_leaks) / 3.0

perm_idx = [2, 1, 0, 5, 4, 3]
P = np.zeros((6, 6))
P[np.arange(6), perm_idx] = 1.0

u_sliced = np.nan_to_num(act_values[:, start_idx:end_idx + 1], nan=0.0)
u_scaled = air_leaks @ P @ (bar2pa * u_sliced)
u_scaled = u_scaled[:, ::stride]

# =====================================================================
# Time-window weights (optional). Uncomment for the MATLAB windowed
# emphasis; leave commented for uniform weighting. Note: the weights apply
# to exp_time, which is ZEROED (starts at 0), so use relative times.
# =====================================================================
# custom_weights = generate_time_window_weights(
#     exp_time, 15.0, 25.0, steepness=3.0, base_weight=0.0, peak_weight=1.0
# )

# =====================================================================
# Robot with NOMINAL (initial-guess) parameters — identical to the
# simulation script; the identification perturbs E, Eta, Rho around these.
# =====================================================================
NOMINAL_E = 1.6464e6
NOMINAL_ETA = 1.0e4
NOMINAL_RHO = 1.104e3
NOMINAL_POI = 0.5
PLA_DENSITY = 1210.0

num_pneumatic_segments = 2
G0 = NOMINAL_E / (2.0 * (1.0 + NOMINAL_POI))

rigid_segment_selector = (True, False, True, False, True)
physical_segment_lengths = jnp.array([41e-3, 180e-3, 27e-3, 180e-3, 6e-3])
physical_segment_radii = 30e-3 * jnp.ones((len(rigid_segment_selector),))
physical_segment_densities = jnp.array(
    [PLA_DENSITY, NOMINAL_RHO, PLA_DENSITY, NOMINAL_RHO, PLA_DENSITY]
)
pcs_segment_counts = (1, 1)

# soft_mask over the 5 PHYSICAL segments = complement of rigid_segment_selector
soft_mask_phys = np.array([not r for r in rigid_segment_selector])  # [F,T,F,T,F]

def update_isupport(robot, p_real):
    """Trace-safe [E, Eta, Rho, Poi] -> updated ISupport.

    Mirrors ISupport.with_params (re-expand params + refresh precomputed
    stiffness/inertia) but builds the new PHYSICAL params with eqx.tree_at
    instead of params.replace(), so there is NO validate() call and the whole
    thing can be traced by autodiff (adam / scipy_jac='autodiff').
    """
    E, eta, rho, poi = p_real[0], p_real[1], p_real[2], p_real[3]
    G = E / (2.0 * (1.0 + poi))
    m = soft_mask_phys
    p = robot.params  # UNEXPANDED physical params (len 5)

    # 1. New physical params, no .replace()/validate()
    new_params = eqx.tree_at(
        lambda pp: (pp.young_modulus, pp.shear_modulus,
                    pp.material_damping_coefficient, pp.density),
        p,
        (jnp.where(m, E,   p.young_modulus),
         jnp.where(m, G,   p.shear_modulus),
         jnp.where(m, eta, p.material_damping_coefficient),
         jnp.where(m, rho, p.density)),
    )

    # 2. Re-expand physical -> PCS params (same call with_params uses)
    pcs_params, _, _, _, _ = _expand_isupport_layout(new_params, robot.structure)

    # 3. Swap in the expanded params + the stored physical params, then
    #    refresh the precomputed stiffness/inertia caches.
    updated = robot._with_pcs_params(pcs_params, stored_params=new_params)
    return updated._with_refreshed_precomputed_matrices()

theta = -jnp.pi / 6.0  # -30 deg experimental offset

params = ISupportParams(
    base_pose=jnp.array([
        jnp.cos(theta / 2), jnp.sin(theta / 2), 0.0, 0.0,
        0.0, 0.0, 0.0,
    ]),
    length=physical_segment_lengths,
    radius=physical_segment_radii,
    density=physical_segment_densities,
    gravity=jnp.array([9.81, 0.0, 0.0]),
    young_modulus=NOMINAL_E * jnp.ones((len(rigid_segment_selector),)),
    shear_modulus=G0 * jnp.ones((len(rigid_segment_selector),)),
    material_damping_coefficient=NOMINAL_ETA
    * jnp.ones((len(rigid_segment_selector),)),
    reference_strain=jnp.tile(
        jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        len(rigid_segment_selector),
    ),
    chamber_inner_radius=6.39e-3 * jnp.ones((num_pneumatic_segments,)),
    chamber_outer_radius=7.79e-3 * jnp.ones((num_pneumatic_segments,)),
    chamber_distance=20e-3 * jnp.ones((num_pneumatic_segments,)),
    chamber_azimuth_angles=jnp.tile(
        2.0 * jnp.pi * jnp.arange(3) / 3 + jnp.pi / 2,
        (num_pneumatic_segments, 1),
    ),
    pcs_segment_lengths=None,
)

robot = ISupport(
    params=params,
    structure=ISupportStructure(
        pcs_segment_counts=pcs_segment_counts,
        rigid_segment_selector=rigid_segment_selector,
    ),
)

# Arc length of the mid cross-section: plate at the end of the middle
# interface (base interface + first pneumatic section + middle interface).
s_mid = float(jnp.sum(physical_segment_lengths[:3]))

# =====================================================================
# Identification
# =====================================================================
identificator = DynamicGVSIdentification(
    robot,
    exp_time,
    poses_window,
    u_scaled,
    normalization=True,
    low_pass=True,
    fc=2.8,
    pi_mask=[1, 1, 1, 0],          # identify E, Eta, Rho; freeze Poi
    mid_idx=4,                     # MATLAB mid_idx = 5 (1-based)
    s_mid=s_mid,
    w_mid=0.0,
    w_tip=1.0,
    # w_time=custom_weights,
    robot_update_fn=update_isupport,
    ode_solver=Tsit5(),
    stepsize_controller=PIDController(rtol=1e-5, atol=1e-7),
    solver_dt=1e-3,
    max_steps=ROLLOUT_MAX_STEPS,   # FINITE: required for reverse-mode AD (adam)
)

# Sanity check: one nominal forward simulation before optimizing.
print("Running nominal forward simulation (also triggers JIT compile)...")
p_tip0, p_mid0, _ = identificator.forward_simulation(
    identificator.p_full_nominal
)
print("nominal resnorm:", identificator.cost_scalar(identificator.pi0))

# =====================================================================
# (A) Confirm parameters reach the dynamics (Case-2 check).
#     A 10x change in E must visibly move the tip; if it doesn't, the
#     robot update isn't refreshing the precomputed stiffness/inertia.
# =====================================================================
p_nom = identificator.p_full_nominal
p_stiff = p_nom * jnp.array([10.0, 1.0, 1.0, 1.0])
tip_nom,   _, _ = identificator.forward_simulation(p_nom)
tip_stiff, _, _ = identificator.forward_simulation(p_stiff)
delta = float(np.max(np.abs(np.asarray(tip_stiff) - np.asarray(tip_nom))))
print(f"[Case-2 probe] max tip change for 10x E: {delta:.3e} m")
if delta <= 1e-6:
    print("  WARNING: tip barely responds to 10x E — update/window "
          "may not be reaching the dynamics.")
else:
    print("  OK: parameters reach the dynamics.")

# =====================================================================
# (B) Autodiff readiness probe (REVERSE mode, for 'adam').
#     This soromox build uses a custom_vjp on its rollout, so FORWARD mode
#     (jacfwd / 'lm') is unavailable. We therefore probe REVERSE mode:
#       - jax.grad on the scalar loss  -> what 'adam' uses;
#       - jax.jacrev on the residual   -> reverse-mode Jacobian, used for
#         the correctness cross-check in (C).
#     A finite result means the trace SUCCEEDS; (C) checks it is CORRECT.
# =====================================================================
def scalar_loss(pi_active):
    # Scalar sum-of-squares of the (non-jitted) residual; pi_active is the
    # ACTIVE normalized parameter vector.
    return jnp.sum(identificator._residual_fn(pi_active) ** 2)

pi0 = identificator.pi0

# 1) reverse-mode gradient of the scalar loss (this is what 'adam' needs)
try:
    g_rev = jax.grad(scalar_loss)(pi0)
    reverse_ok = bool(jnp.all(jnp.isfinite(g_rev)))
    print("[AD probe] reverse-mode grad OK:", np.asarray(g_rev),
          "finite:", reverse_ok)
except Exception as e:
    print("[AD probe] reverse-mode grad FAILED:",
          type(e).__name__, str(e)[:200])
    reverse_ok = False

# 2) reverse-mode Jacobian of the residual (jacrev) for the cross-check
try:
    J = jax.jacrev(identificator._residual_fn)(pi0)
    jacrev_ok = bool(jnp.all(jnp.isfinite(J)))
    print("[AD probe] reverse-mode jacrev OK, shape:", J.shape,
          "finite:", jacrev_ok)
except Exception as e:
    print("[AD probe] reverse-mode jacrev FAILED:",
          type(e).__name__, str(e)[:200])
    jacrev_ok = False
    J = None

# =====================================================================
# (C) Numerical cross-check: reverse-mode AD Jacobian vs central finite
#     differences. Tracing successfully (B) is necessary but NOT
#     sufficient — a finite gradient can still be wrong if something in
#     the rollout differentiates incorrectly. This is the test that tells
#     you the autodiff is TRUSTWORTHY.
#       small relative difference (<< 1, e.g. < 1e-3) -> AD is correct,
#       use solver='adam'. Large -> stay on solver='scipy' (FD).
# =====================================================================
if jacrev_ok and J is not None:
    print("[AD vs FD] computing central finite-difference Jacobian "
          "(this runs a few extra forward sims)...")
    J_fd = _central_fd_jacobian(
        lambda x: identificator._residual_fn(jnp.asarray(x)),
        pi0, step=1e-2,
    )
    J_ad = np.asarray(J)
    rel = np.linalg.norm(J_ad - J_fd) / (np.linalg.norm(J_fd) + 1e-30)
    print(f"[AD vs FD] relative difference: {rel:.2e}  (want << 1, e.g. < 1e-3)")
    autodiff_trustworthy = rel < 1e-3
    print("[AD vs FD] autodiff trustworthy:", autodiff_trustworthy,
          "-> recommended solver:",
          "'adam' (reverse-mode autodiff)" if autodiff_trustworthy
          else "'scipy' (FD)")
else:
    autodiff_trustworthy = False
    print("[AD vs FD] skipped (reverse-mode AD not available) "
          "-> use solver='scipy' (FD)")

# =====================================================================
# Solver choice
# ---------------------------------------------------------------------
# "scipy"  : TRF least squares + central finite differences. Only needs
#            forward simulations -> always works; closest to lsqnonlin
#            ('central', DiffMinChange 1e-2).
# "adam"   : optax Adam on the scalar loss, REVERSE-mode gradient.
#            Requires a finite max_steps on the rollout (set above).
# "lm"     : optimistix Levenberg-Marquardt with a FORWARD-mode (jacfwd)
#            Jacobian -> NOT usable with this soromox build (custom_vjp).
#
# Auto-pick: use 'adam' if the reverse-mode AD Jacobian matched FD, else
# fall back to 'scipy'. Override by hand if you prefer.
# =====================================================================
chosen_solver = "adam" if autodiff_trustworthy else "scipy"
print("Using solver:", chosen_solver)

if chosen_solver == "adam":
    pi_hat, info, result = identificator.solve(
        solver="adam",
        adam_lr=1e-2,
        adam_steps=100,
        adam_clip=1.0,
        bounded=True,
        compute_jacobian=True,   # fills condition number / correlations
    )
else:
    pi_hat, info, result = identificator.solve(
        solver="scipy",
        scipy_jac="3-point",
        scipy_diff_step=1e-2,
        compute_jacobian=False,   # scipy already returns it in info["jacobian"]
    )

print("Identified [E; Eta; Rho; Poi]:", pi_hat)
if "condition_number" in info:
    print("Jacobian condition number:", info["condition_number"])
    print("Parameter correlation:\n", info.get("param_correlation"))

np.save(os.path.join(RESULTS_DIR, "pi_hat.npy"), pi_hat)

# =====================================================================
# Plots and video
# =====================================================================
identificator.show_solution(pi_hat, sim_tip=info["p_sim_tip"],
                            sim_mid=info["p_sim_mid"],
                            save_dir=RESULTS_DIR, show=False)

anim = identificator.show_video(
    pi_hat, result=result, step=10, frame_rate=10,
    save_video=False,
    file_name=os.path.join(RESULTS_DIR, "identification_video.mp4"),
)