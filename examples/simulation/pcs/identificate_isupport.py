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
* gradients through the rollout come from autodiff (see solver choice
  below) instead of parallel finite differences.
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
)

RESULTS_DIR = "amisupport_dataset/results_identification"
os.makedirs(RESULTS_DIR, exist_ok=True)

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
# Time-window weights (30 s -> 40 s, steepness 3, base 0, peak 1)
# =====================================================================
# custom_weights = generate_time_window_weights(
#     exp_time, 30.0, 40.0, steepness=3.0, base_weight=0.0, peak_weight=1.0
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
    E, eta, rho, poi = p_real[0], p_real[1], p_real[2], p_real[3]
    G = E / (2.0 * (1.0 + poi))

    p = robot.params                      # UNEXPANDED physical params (len 5)
    m = jnp.asarray(soft_mask_phys)
    return robot.update_params(
        young_modulus=jnp.where(m, E,   p.young_modulus),
        shear_modulus=jnp.where(m, G,   p.shear_modulus),
        material_damping_coefficient=jnp.where(m, eta, p.material_damping_coefficient),
        density=jnp.where(m, rho, p.density),
    )

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
    max_steps=None,   # set e.g. 16**5 if reverse-mode AD complains
)

# Sanity check: one nominal forward simulation before optimizing.
print("Running nominal forward simulation (also triggers JIT compile)...")
p_tip0, p_mid0, _ = identificator.forward_simulation(
    identificator.p_full_nominal
)
print("nominal resnorm:", identificator.cost_scalar(identificator.pi0))

# --- Solver choice ----------------------------------------------------
# "scipy"  : TRF least squares + central finite differences. Only needs
#            forward simulations -> always works; closest to lsqnonlin
#            ('central', DiffMinChange 1e-2). Use scipy_jac="autodiff"
#            to switch to a forward-mode AD Jacobian once verified.
# "lm"     : optimistix Levenberg-Marquardt, autodiff Jacobian.
# "adam"   : optax Adam on the scalar loss, reverse-mode gradient
#            (requires finite max_steps in the rollout for diffrax).
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
                            save_dir=RESULTS_DIR, show=True)

identificator.show_video(
    pi_hat, result=result, step=10, frame_rate=10,
    save_video=False,
    file_name=os.path.join(RESULTS_DIR, "identification_video.mp4"),
)