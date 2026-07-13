"""Dynamic GVS parameter identification for soromox.

Python/JAX port of the MATLAB ``GVS_Identification`` class (SoRoSim).

Pipeline (identical to the MATLAB class):

    theta_active (normalized) --expand--> [E, Eta, Rho, Poi] (real units)
        --update robot--> functional copy of the ISupport model
        --rollout dynamics (with optional 1st-order low-pass on the input,
          integrated together with the mechanics, as in FD_filtered.m /
          simulate_robot_lp.m)-->
        q(t) --forward kinematics--> mid / tip positions
        --weighted residuals vs. Vicon data--> optimizer

What is *new* w.r.t. MATLAB, borrowed from the soromox static-identification
example:

* the robot is an Equinox pytree, so parameter updates are functional
  (``eqx.tree_at``) and the whole simulation is differentiable / jittable;
* gradients can come from autodiff (forward-mode Jacobian is ideal here:
  only 3-4 parameters vs. thousands of residuals) instead of MATLAB's
  parallel finite differences;
* three interchangeable solvers:
    - ``"lm"``     : optimistix Levenberg-Marquardt on the residual vector
                     (closest in spirit to ``lsqnonlin``), autodiff Jacobian;
    - ``"adam"``   : optax Adam on the scalar sum-of-squares (same pattern
                     as the static-identification script), reverse-mode grad;
    - ``"scipy"``  : ``scipy.optimize.least_squares`` (TRF) with native box
                     bounds and either central finite differences
                     (== MATLAB 'FiniteDifferenceType','central') or an
                     autodiff Jacobian. This path only requires *forward*
                     simulations, so it always works even if reverse/forward
                     AD through the soromox rollout is not supported by the
                     installed diffrax adjoint.

Parameter vector, masking, normalization, bounds, and the residual
definition (sqrt-weighted componentwise position errors at mid and tip,
stacked as [res_mid(:); res_tip(:)]) follow the MATLAB class one-to-one.
"""

from __future__ import annotations

import time as _walltime
from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as onp

try:  # optional deps, only needed for the corresponding solver
    import optax
except ImportError:  # pragma: no cover
    optax = None
try:
    import optimistix as optx
except ImportError:  # pragma: no cover
    optx = None

from diffrax import PIDController, Tsit5

from soromox.systems import SystemState

jax.config.update("jax_enable_x64", True)

PARAM_NAMES = ("E", "Eta", "Rho", "Poi")


# =====================================================================
# Recorded-input controller with optional first-order low-pass filter.
# Port of FD_filtered.m / simulate_robot_lp.m: the filter state u_filt is
# carried as `control_state` and integrated by the same adaptive solver,
#       tau * d(u_filt)/dt = u_raw(t) - u_filt,   tau = 1 / (2*pi*fc).
# The input passed to the identification class is assumed to be ALREADY
# scaled/permuted (exactly like `obj.input` in the MATLAB class: scaling
# with chamber surface / air leaks / permutation is done by the caller).
# =====================================================================
class RecordedInputController(eqx.Module):
    act_time: jax.Array            # (M,)
    act_values: jax.Array          # (n_act, M)
    tau: jax.Array                 # ()
    low_pass: bool = eqx.field(static=True)

    def __call__(self, state: SystemState):
        t = state.t
        u_raw = jax.vmap(lambda row: jnp.interp(t, self.act_time, row))(
            self.act_values
        )
        if self.low_pass:
            u_filt = state.control_state
            du_dt = (u_raw - u_filt) / self.tau
            return u_filt, du_dt
        # No filtering: pass the raw command through; keep the (dummy)
        # control_state constant so the augmented ODE stays well defined.
        return u_raw, jnp.zeros_like(state.control_state)


# =====================================================================
# Functional counterpart of update_robot.m.
# params = [E; Eta; Rho; Poi] -> new robot pytree. Only the *soft*
# (pneumatic) segments are touched, exactly like VLinks(2) in MATLAB;
# the Poisson ratio enters through the shear modulus G = E / (2*(1+Poi)).
# =====================================================================
def default_update_robot(robot, p_real: jax.Array, soft_mask: jax.Array):
    """Return a copy of ``robot`` with [E, Eta, Rho, Poi] applied to the
    segments selected by ``soft_mask`` (bool array over the physical
    segments of ``robot.params``)."""
    E, eta, rho, poi = p_real[0], p_real[1], p_real[2], p_real[3]
    G = E / (2.0 * (1.0 + poi))

    p = robot.params
    young = jnp.where(soft_mask, E, p.young_modulus)
    shear = jnp.where(soft_mask, G, p.shear_modulus)
    damping = jnp.where(soft_mask, eta, p.material_damping_coefficient)
    density = jnp.where(soft_mask, rho, p.density)

    new_params = eqx.tree_at(
        lambda pp: (
            pp.young_modulus,
            pp.shear_modulus,
            pp.material_damping_coefficient,
            pp.density,
        ),
        p,
        (young, shear, damping, density),
    )
    return eqx.tree_at(lambda r: r.params, robot, new_params)


def _central_fd_jacobian(fun, x, step: float):
    """Central finite-difference Jacobian (MATLAB 'central' + DiffMinChange).

    Only forward evaluations of ``fun`` are needed — guaranteed fallback
    when autodiff through the ODE rollout is unavailable."""
    x = onp.asarray(x, dtype=float)
    cols = []
    for k in range(x.size):
        h = max(step, step * abs(x[k]))
        xp, xm = x.copy(), x.copy()
        xp[k] += h
        xm[k] -= h
        cols.append((onp.asarray(fun(xp)) - onp.asarray(fun(xm))) / (2.0 * h))
    return onp.stack(cols, axis=1)


# =====================================================================
# Main class
# =====================================================================
class DynamicGVSIdentification:
    """Dynamic identification of [E, Eta, Rho, Poi] for a soromox ISupport
    (or any soromox system exposing the same ``rollout_closed_loop_to`` /
    ``forward_kinematics(q, s)`` interface).

    Parameters
    ----------
    robot :
        soromox system built with the *nominal* (initial-guess) parameters.
    time : (M,) array
        Experimental time grid, starting at the window origin (t[0] may be
        nonzero; the rollout starts at t[0]).
    poses_data : (7, N, M) array
        Vicon poses [pos(3); quat(4)] for the N tracked cross-sections.
        Column ``mid_idx`` is the mid cross-section, the last column is the
        tip — same convention as the MATLAB class.
    actuation_input : (n_act, M) array
        Already-scaled actuation samples on ``time`` (chamber pressures in
        Pa for ISupport). Interpolated linearly in time, like
        ``obj.u_fh = @(t) interp1(...)``.
    normalization : bool
        If True, the optimizer works on parameters scaled by
        diag([1e6, 1e4, 1e3, 1]) — identical to the MATLAB class.
    low_pass, fc :
        First-order low-pass on the input, integrated with the dynamics.
        ``tau = 1/(2*pi*fc)``. The filter state is initialized at the raw
        command at t[0] (matches ``simulate_robot_lp``).
    pi_mask : length-4 0/1 sequence over [E, Eta, Rho, Poi]
        1 -> identified, 0 -> frozen at nominal.
    mid_idx : int
        Column of ``poses_data`` used as mid cross-section (MATLAB default
        column 5 -> Python index 4).
    s_mid, s_tip : float
        Arc lengths [m] at which the simulated mid / tip frames are
        evaluated by forward kinematics (replaces ``amISupport_FK``).
        ``s_tip`` defaults to the total length; ``s_mid`` must be given if
        ``w_mid > 0``.
    w_mid, w_tip : float
        Block weights; applied as sqrt(w) on each residual block.
    w_time : (M,) array or None
        Per-sample weights, normalized to sum to 1 (uniform if None).
    soft_segment_mask : bool array or None
        Which physical segments receive the identified parameters. Defaults
        to the complement of ``robot.structure.rigid_segment_selector``.
    robot_update_fn : callable(robot, p_real) -> robot, optional
        Override if your soromox version caches quantities that
        ``eqx.tree_at`` on ``robot.params`` would not refresh.
    ode_solver, stepsize_controller, solver_dt, max_steps :
        Passed to ``rollout_closed_loop_to``. Note: reverse-mode autodiff
        through diffrax typically requires a finite ``max_steps``.
    lb, ub : length-4 sequences
        Box bounds in *normalized* units (MATLAB defaults:
        lb = 1e-3, ub = [1e2, 1e2, 1e2, 0.5]).
    """

    def __init__(
        self,
        robot,
        time,
        poses_data,
        actuation_input,
        *,
        normalization: bool = True,
        low_pass: bool = True,
        fc: float = 1.0,
        pi_mask: Sequence[int] = (1, 1, 1, 1),
        mid_idx: int = 4,
        s_mid: Optional[float] = None,
        s_tip: Optional[float] = None,
        w_mid: float = 1.0,
        w_tip: float = 1.0,
        w_time=None,
        soft_segment_mask=None,
        robot_update_fn: Optional[Callable] = None,
        ode_solver=None,
        stepsize_controller=None,
        solver_dt: float = 1e-3,
        max_steps: Optional[int] = None,
        lb: Sequence[float] = (1e-3, 1e-3, 1e-3, 1e-3),
        ub: Sequence[float] = (1e2, 1e2, 1e2, 5e-1),
    ):
        self.robot = robot
        self.time = jnp.asarray(time, dtype=jnp.float64).ravel()
        self.M = int(self.time.shape[0])
        self.tf = float(self.time[-1])
        self.input = jnp.asarray(actuation_input, dtype=jnp.float64)
        assert self.input.shape[1] == self.M, "input must be (n_act, M)"

        poses_data = jnp.asarray(poses_data, dtype=jnp.float64)
        assert poses_data.ndim == 3 and poses_data.shape[0] >= 7, \
            "poses_data must be (7, N, M)"
        self.poses_data = poses_data
        self.mid_idx = int(mid_idx)

        # --- weights (as in MATLAB: sqrt(w_block * w_time) per component) --
        assert w_mid >= 0.0 and w_tip >= 0.0, "w_mid, w_tip must be >= 0"
        self.w_mid, self.w_tip = float(w_mid), float(w_tip)
        if w_time is None:
            w_time = jnp.ones((self.M,)) / self.M
        else:
            w_time = jnp.asarray(w_time, dtype=jnp.float64).ravel()
            assert w_time.shape[0] == self.M, "w_time must have M samples"
            assert bool(jnp.all(w_time >= 0)), "time weights must be >= 0"
            w_time = w_time / jnp.sum(w_time)
        self.w_time = w_time

        # --- experimental targets + NaN masking (Vicon dropouts) ----------
        p_exp_tip = poses_data[0:3, -1, :]
        p_exp_mid = poses_data[0:3, self.mid_idx, :]
        tip_valid = jnp.all(jnp.isfinite(p_exp_tip), axis=0)  # (M,)
        mid_valid = jnp.all(jnp.isfinite(p_exp_mid), axis=0)
        self.p_exp_tip = jnp.nan_to_num(p_exp_tip)
        self.p_exp_mid = jnp.nan_to_num(p_exp_mid)
        # sqrt-weights folded with validity masks -> (1, M), broadcast to 3xM
        self._sw_tip = jnp.sqrt(self.w_tip * self.w_time) * tip_valid
        self._sw_mid = jnp.sqrt(self.w_mid * self.w_time) * mid_valid

        # --- low-pass filtering -------------------------------------------
        self.low_pass = bool(low_pass)
        self.fc = float(fc)
        self.tau = 1.0 / (2.0 * jnp.pi * self.fc)

        # --- parameter masking / normalization / bounds -------------------
        pi_mask = onp.asarray(pi_mask).astype(bool).ravel()
        assert pi_mask.size == 4, "pi_mask must have 4 entries [E,Eta,Rho,Poi]"
        assert pi_mask.any(), "pi_mask must keep at least one parameter active"
        self.pi_mask = pi_mask
        # Static integer indices of the active params (jit-friendly scatter)
        self._active_idx = onp.flatnonzero(pi_mask)
        self.normalization = bool(normalization)
        self.norm_vec = (
            jnp.array([1e6, 1e4, 1e3, 1e0]) if self.normalization
            else jnp.ones((4,))
        )
        self.lb_full = jnp.asarray(lb, dtype=jnp.float64)
        self.ub_full = jnp.asarray(ub, dtype=jnp.float64)

        # --- geometry: where to evaluate FK -------------------------------
        total_L = float(jnp.sum(robot.L))
        self.s_tip = float(s_tip) if s_tip is not None else total_L
        if s_mid is None:
            if self.w_mid > 0.0:
                raise ValueError(
                    "w_mid > 0 requires s_mid (arc length of the mid "
                    "cross-section, e.g. sum of segment lengths up to the "
                    "middle interface)."
                )
            s_mid = 0.5 * total_L  # placeholder, weight is zero anyway
        self.s_mid = float(s_mid)

        # --- which segments receive the identified parameters -------------
        if soft_segment_mask is None:
            rigid = onp.asarray(robot.structure.rigid_segment_selector)
            soft_segment_mask = ~rigid
        self.soft_mask = jnp.asarray(soft_segment_mask, dtype=bool)

        self._user_update_fn = robot_update_fn

        # --- nominal parameters and initial guess (normalized, active) ----
        self.p_full_nominal = self.extract_params()
        pi0_full = self.p_full_nominal / self.norm_vec
        self.pi0 = pi0_full[self._active_idx]

        # --- ODE solver setup ---------------------------------------------
        self.ode_solver = ode_solver if ode_solver is not None else Tsit5()
        self.stepsize_controller = (
            stepsize_controller
            if stepsize_controller is not None
            else PIDController(rtol=1e-5, atol=1e-7)
        )
        self.solver_dt = float(solver_dt)
        self.max_steps = max_steps
        self._save_dt = float(self.time[1] - self.time[0])

        # --- jitted pure functions ----------------------------------------
        self._sim_fn = eqx.filter_jit(self._build_sim_fn())
        self._residual_fn = eqx.filter_jit(self._build_residual_fn())
        self._loss_fn = lambda pi: jnp.sum(self._residual_fn(pi) ** 2)

    # ------------------------------------------------------------------
    # Parameter bookkeeping (extract_params / expand_params of MATLAB)
    # ------------------------------------------------------------------
    def extract_params(self) -> jax.Array:
        """Full [E; Eta; Rho; Poi] (real units) of the soft segments of the
        current nominal robot. Poi is recovered from E and G."""
        p = self.robot.params
        idx = int(onp.flatnonzero(onp.asarray(self.soft_mask))[0])
        E = p.young_modulus[idx]
        G = p.shear_modulus[idx]
        eta = p.material_damping_coefficient[idx]
        rho = p.density[idx]
        poi = E / (2.0 * G) - 1.0
        return jnp.array([E, eta, rho, poi])

    def expand_params(self, pi_active: jax.Array) -> jax.Array:
        """ACTIVE normalized params -> FULL real-unit [E; Eta; Rho; Poi]."""
        pi_full = self.p_full_nominal / self.norm_vec
        pi_full = pi_full.at[self._active_idx].set(pi_active)
        return self.norm_vec * pi_full

    def _update_robot(self, p_real: jax.Array):
        if self._user_update_fn is not None:
            return self._user_update_fn(self.robot, p_real)
        return default_update_robot(self.robot, p_real, self.soft_mask)

    # ------------------------------------------------------------------
    # Forward simulation (forward_simulation of MATLAB)
    # ------------------------------------------------------------------
    def _build_sim_fn(self):
        time, u_samples = self.time, self.input
        tau, low_pass = self.tau, self.low_pass
        s_mid, s_tip = self.s_mid, self.s_tip
        solver, controller_ss = self.ode_solver, self.stepsize_controller
        solver_dt, save_dt, max_steps = self.solver_dt, self._save_dt, self.max_steps

        def simulate(p_real):
            robot_p = self._update_robot(p_real)

            ctrl = RecordedInputController(
                act_time=time, act_values=u_samples,
                tau=jnp.asarray(tau), low_pass=low_pass,
            )
            # Filter initialized at the raw command at the window start,
            # exactly like simulate_robot_lp (uf0 = u_fh(t0)).
            u_f0 = u_samples[:, 0] if low_pass else jnp.zeros(u_samples.shape[0])

            n_act = robot_p.num_actuators
            # x0 = zeros(2*ndof, 1): rest configuration, zero velocity.
            n_q = _num_dof(robot_p)
            state0 = SystemState(
                t=time[0],
                y=jnp.zeros((2 * n_q,)),
                u=jnp.zeros((n_act,)),
                control_state=u_f0,
            )
            traj = robot_p.rollout_closed_loop_to(
                initial_state=state0,
                controller=ctrl,
                t1=time[-1],
                solver_dt=solver_dt,
                save_dt=save_dt,
                save_ts=time,
                solver=solver,
                stepsize_controller=controller_ss,
                max_steps=max_steps,
            )
            q_ts, _ = jnp.split(traj.y, 2, axis=1)  # (M, n_q)

            fk_mid = jax.vmap(lambda q: robot_p.forward_kinematics(q, s_mid))(q_ts)
            fk_tip = jax.vmap(lambda q: robot_p.forward_kinematics(q, s_tip))(q_ts)
            p_sim_mid = fk_mid[:, 0:3, 3].T  # (3, M)
            p_sim_tip = fk_tip[:, 0:3, 3].T
            return p_sim_mid, p_sim_tip, q_ts

        return simulate

    def forward_simulation(self, p_real):
        """Simulate with FULL real-unit params. Returns
        (p_sim_tip (3,M), p_sim_mid (3,M), q_ts (M, n_q)) — the MATLAB
        method returned (p_sim_tip, p_sim_mid, result)."""
        p_sim_mid, p_sim_tip, q_ts = self._sim_fn(jnp.asarray(p_real))
        return p_sim_tip, p_sim_mid, q_ts

    # ------------------------------------------------------------------
    # Residuals / cost (cost_function / cost_scalar of MATLAB)
    # ------------------------------------------------------------------
    def _build_residual_fn(self):
        sim = self._build_sim_fn()
        p_exp_mid, p_exp_tip = self.p_exp_mid, self.p_exp_tip
        sw_mid, sw_tip = self._sw_mid, self._sw_tip

        def residuals(pi_active):
            p_real = self.expand_params(pi_active)
            p_sim_mid, p_sim_tip, _ = sim(p_real)
            res_mid = (p_exp_mid - p_sim_mid) * sw_mid  # (3, M), NaNs masked
            res_tip = (p_exp_tip - p_sim_tip) * sw_tip
            # MATLAB: cost = [res_mid(:); res_tip(:)]
            return jnp.concatenate([res_mid.ravel(), res_tip.ravel()])

        return residuals

    def cost_function(self, pi_active):
        """Residual vector (6*M,) — lsqnonlin form."""
        return self._residual_fn(jnp.asarray(pi_active))

    def cost_scalar(self, pi_active):
        """Sum of squared residuals — particleswarm/Adam form."""
        return float(self._loss_fn(jnp.asarray(pi_active)))

    # ------------------------------------------------------------------
    # tanh box-bound reparameterization (from the static-identification
    # script): normalized param <-> unconstrained "raw" variable.
    # ------------------------------------------------------------------
    def _bound_maps(self):
        lb = self.lb_full[self._active_idx]
        ub = self.ub_full[self._active_idx]
        mu, r = 0.5 * (lb + ub), 0.5 * (ub - lb)

        def to_box(raw):
            return mu + r * jnp.tanh(raw)

        def to_raw(x, eps=1e-9):
            z = jnp.clip((x - mu) / r, -1 + eps, 1 - eps)
            return jnp.arctanh(z)

        return to_box, to_raw

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(
        self,
        *,
        solver: str = "lm",
        pi0=None,
        bounded: bool = True,
        verbose: bool = True,
        # lm (optimistix)
        lm_rtol: float = 1e-8,
        lm_atol: float = 1e-8,
        lm_max_steps: int = 256,
        # adam (optax)
        adam_lr: float = 1e-2,
        adam_steps: int = 100,
        adam_clip: float = 1.0,
        # scipy
        scipy_jac: str = "3-point",   # "3-point" (central FD) | "autodiff"
        scipy_diff_step: float = 1e-2,  # ~ MATLAB DiffMinChange
        scipy_kwargs: Optional[dict] = None,
        compute_jacobian: bool = False,
    ):
        """Run the identification. Returns (pi_hat, info, result) like the
        MATLAB ``solve``: pi_hat is the FULL real-unit [E; Eta; Rho; Poi],
        info collects solver diagnostics, result = (p_sim_tip, p_sim_mid,
        q_ts) of the final simulation."""
        pi0 = self.pi0 if pi0 is None else jnp.asarray(pi0).ravel()
        names = [n for n, m in zip(PARAM_NAMES, self.pi_mask) if m]
        if verbose:
            print("Start Optimization")
            print("Active parameters:", ", ".join(names))
            print("Solver:", solver)
            print("pi0 (normalized):", onp.asarray(pi0))

        info = {
            "solver": solver,
            "active_names": names,
            "pi_mask": self.pi_mask.copy(),
            "lb": onp.asarray(self.lb_full[self._active_idx]),
            "ub": onp.asarray(self.ub_full[self._active_idx]),
            "pi0": onp.asarray(pi0),
        }

        t_start = _walltime.time()
        if solver == "lm":
            pi_active_hat = self._solve_lm(
                pi0, bounded, lm_rtol, lm_atol, lm_max_steps, info, verbose
            )
        elif solver == "adam":
            pi_active_hat = self._solve_adam(
                pi0, bounded, adam_lr, adam_steps, adam_clip, info, verbose
            )
        elif solver == "scipy":
            pi_active_hat = self._solve_scipy(
                pi0, scipy_jac, scipy_diff_step, scipy_kwargs, info, verbose
            )
        else:
            raise ValueError("solver must be 'lm', 'adam', or 'scipy'")
        info["wall_time_s"] = _walltime.time() - t_start

        pi_star = self.expand_params(pi_active_hat)
        info["pi_active_hat"] = onp.asarray(pi_active_hat)
        info["pi_star"] = onp.asarray(pi_star)
        info["resnorm"] = float(self._loss_fn(pi_active_hat))

        if compute_jacobian:
            info["jacobian"] = self.compute_jacobian(pi_active_hat)
            info.update(_jacobian_diagnostics(info["jacobian"]))

        if verbose:
            print("Identified Parameters [E; Eta; Rho; Poi]:")
            print(onp.asarray(pi_star))
            print(f"resnorm = {info['resnorm']:.6e} "
                  f"({info['wall_time_s']:.1f} s)")

        # Final simulation, kept so downstream calls don't re-simulate.
        p_sim_tip, p_sim_mid, q_ts = self.forward_simulation(pi_star)
        info["p_sim_tip"] = onp.asarray(p_sim_tip)
        info["p_sim_mid"] = onp.asarray(p_sim_mid)
        self.updated_robot = self._update_robot(pi_star)
        result = {"t": onp.asarray(self.time), "q_ts": onp.asarray(q_ts),
                  "p_sim_tip": onp.asarray(p_sim_tip),
                  "p_sim_mid": onp.asarray(p_sim_mid)}
        return onp.asarray(pi_star), info, result

    # --- individual solvers -------------------------------------------
    def _solve_lm(self, pi0, bounded, rtol, atol, max_steps, info, verbose):
        if optx is None:
            raise ImportError("optimistix is required for solver='lm'")
        res_fn = self._residual_fn
        if bounded:
            to_box, to_raw = self._bound_maps()
            fn = lambda raw, args: res_fn(to_box(raw))
            y0 = to_raw(pi0)
        else:
            fn = lambda y, args: res_fn(y)
            y0 = pi0
        sol = optx.least_squares(
            fn,
            optx.LevenbergMarquardt(rtol=rtol, atol=atol),
            y0,
            max_steps=max_steps,
            throw=False,
        )
        info["optimistix_result"] = sol.result
        if verbose:
            print("optimistix result:", sol.result)
        return to_box(sol.value) if bounded else sol.value

    def _solve_adam(self, pi0, bounded, lr, steps, clip, info, verbose):
        if optax is None:
            raise ImportError("optax is required for solver='adam'")
        res_fn = self._residual_fn
        if bounded:
            to_box, to_raw = self._bound_maps()
            params = to_raw(pi0)
            loss = lambda raw: jnp.sum(res_fn(to_box(raw)) ** 2)
        else:
            to_box = lambda x: x
            params = pi0
            loss = lambda x: jnp.sum(res_fn(x) ** 2)

        loss_and_grad = eqx.filter_jit(jax.value_and_grad(loss))
        opt = optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))
        opt_state = opt.init(params)

        best_loss, best_params, history = jnp.inf, params, []
        for step in range(steps):
            loss_val, grads = loss_and_grad(params)
            history.append(float(loss_val))
            if float(loss_val) < float(best_loss):
                best_loss, best_params = loss_val, params
            if verbose:
                print(f"[step {step:03d}] loss={float(loss_val):.6e}  "
                      f"pi={onp.asarray(to_box(params))}")
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
        info["loss_history"] = history
        info["best_loss"] = float(best_loss)
        return to_box(best_params)

    def _solve_scipy(self, pi0, jac, diff_step, extra, info, verbose):
        from scipy.optimize import least_squares

        res_np = lambda x: onp.asarray(self._residual_fn(jnp.asarray(x)))
        lb = onp.asarray(self.lb_full[self._active_idx])
        ub = onp.asarray(self.ub_full[self._active_idx])

        if jac == "autodiff":
            jac_fn = eqx.filter_jit(jax.jacfwd(self._residual_fn))
            jac_arg = lambda x: onp.asarray(jac_fn(jnp.asarray(x)))
        else:
            jac_arg = jac  # "2-point" / "3-point" handled by scipy

        kwargs = dict(
            method="trf",
            bounds=(lb, ub),
            diff_step=diff_step,
            x_scale="jac",
            verbose=2 if verbose else 0,
        )
        if extra:
            kwargs.update(extra)
        sol = least_squares(res_np, onp.asarray(pi0), jac=jac_arg, **kwargs)
        info["scipy_result"] = sol
        info["jacobian"] = sol.jac
        info.update(_jacobian_diagnostics(sol.jac))
        return jnp.asarray(sol.x)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def compute_jacobian(self, pi_active, fd_step: float = 1e-2):
        """Jacobian of the residuals at ``pi_active``. Tries forward-mode
        autodiff (cheap: n_params directional derivatives), falls back to
        central finite differences."""
        pi_active = jnp.asarray(pi_active)
        try:
            J = jax.jacfwd(self._residual_fn)(pi_active)
            return onp.asarray(J)
        except Exception as err:  # AD through the rollout unsupported
            print(f"[compute_jacobian] autodiff failed ({type(err).__name__}: "
                  f"{err}); falling back to central finite differences.")
            return _central_fd_jacobian(
                lambda x: self._residual_fn(jnp.asarray(x)), pi_active, fd_step
            )

    # ------------------------------------------------------------------
    # Plots (show_solution of MATLAB)
    # ------------------------------------------------------------------
    def show_solution(self, pi_star, *, sim_tip=None, sim_mid=None,
                      save_dir=None, show=True):
        import matplotlib.pyplot as plt

        if sim_tip is None or sim_mid is None:
            sim_tip, sim_mid, _ = self.forward_simulation(pi_star)
        sim_tip, sim_mid = onp.asarray(sim_tip), onp.asarray(sim_mid)
        t = onp.asarray(self.time)
        exp_tip = onp.asarray(self.poses_data[0:3, -1, :])
        exp_mid = onp.asarray(self.poses_data[0:3, self.mid_idx, :])

        figs = []
        for name, sim, exp in (("Tip", sim_tip, exp_tip),
                               ("Mid", sim_mid, exp_mid)):
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            fig.suptitle(f"{name} position")
            for j, lab in enumerate(("x", "y", "z")):
                axes[j].plot(t, sim[j], lw=2.5, label="Sim. Data")
                axes[j].plot(t, exp[j], "--", lw=1.8, label="Exp. Data")
                axes[j].set_ylabel(f"${lab}$ [m]")
                axes[j].grid(True)
                axes[j].legend()
            axes[-1].set_xlabel("Time [s]")
            fig.tight_layout()
            figs.append(fig)

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(*sim_tip, lw=2.5, label="Tip Sim.")
        ax.plot(*exp_tip, "--", lw=1.8, label="Tip Exp.")
        ax.plot(*sim_mid, lw=2.5, label="Mid Sim.")
        ax.plot(*exp_mid, "--", lw=1.8, label="Mid Exp.")
        ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$y$ [m]"); ax.set_zlabel("$z$ [m]")
        ax.legend(); ax.set_box_aspect([1, 1, 1])
        figs.append(fig)

        if save_dir is not None:
            import os
            os.makedirs(save_dir, exist_ok=True)
            for k, f in enumerate(figs):
                f.savefig(f"{save_dir}/identification_fig_{k}.png",
                          dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        return figs

    # ------------------------------------------------------------------
    # Animation (show_video of MATLAB) — backbone curve vs. Vicon frames.
    # For a fancier rendering, use soromox's MatplotlibRenderer /
    # ISupportViserRenderer with result["q_ts"].
    # ------------------------------------------------------------------
    def show_video(self, pi_star, *, result=None, step=10, num_points=40,
                   frame_rate=10, save_video=False,
                   file_name="identification_video.mp4"):
        import matplotlib.pyplot as plt
        from matplotlib import animation

        if result is None:
            _, _, q_ts = self.forward_simulation(pi_star)
        else:
            q_ts = jnp.asarray(result["q_ts"])
        robot_p = self._update_robot(jnp.asarray(pi_star))

        s_ps = jnp.linspace(0.0, float(jnp.sum(robot_p.L)), num_points)
        fk_all = jax.vmap(
            lambda q: jax.vmap(lambda s: robot_p.forward_kinematics(q, s))(s_ps)
        )(q_ts[::step])                       # (F, num_points, 4, 4)
        curves = onp.asarray(fk_all[:, :, 0:3, 3])
        exp = onp.asarray(self.poses_data[0:3, :, ::step])   # (3, N, F)
        t_sub = onp.asarray(self.time[::step])

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        lo = onp.nanmin(curves.reshape(-1, 3), axis=0) - 0.05
        hi = onp.nanmax(curves.reshape(-1, 3), axis=0) + 0.05

        (line,) = ax.plot([], [], [], lw=3, label="Sim backbone")
        scat = ax.scatter([], [], [], c="k", marker="o", label="Vicon")

        def init():
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
            ax.view_init(elev=90, azim=90)
            ax.legend()
            return line, scat

        def update(f):
            c = curves[f]
            line.set_data(c[:, 0], c[:, 1]); line.set_3d_properties(c[:, 2])
            pts = exp[:, :, f]
            scat._offsets3d = (pts[0], pts[1], pts[2])
            ax.set_title(f"t = {t_sub[f]:.2f} s")
            return line, scat

        ani = animation.FuncAnimation(
            fig, update, frames=curves.shape[0], init_func=init,
            interval=1000 / frame_rate, blit=False,
        )
        if save_video:
            ani.save(file_name, fps=frame_rate, dpi=150)
            print("Video saved to", file_name)
        plt.show()
        return ani


# =====================================================================
# Helpers
# =====================================================================
def _num_dof(robot) -> int:
    """Number of generalized coordinates q of a soromox system."""
    for attr in ("num_dof", "ndof", "n_dof"):
        if hasattr(robot, attr):
            return int(getattr(robot, attr))
    if hasattr(robot, "dofs_per_segment"):
        return int(onp.sum(onp.asarray(robot.dofs_per_segment)))
    # ISupport fallback: 6 strains per PCS segment
    if hasattr(robot, "structure") and hasattr(robot.structure,
                                               "pcs_segment_counts"):
        return 6 * int(sum(robot.structure.pcs_segment_counts))
    raise AttributeError("Could not infer the number of DOFs of the robot; "
                         "please add the correct attribute in _num_dof().")


def _jacobian_diagnostics(J) -> dict:
    """SVD-based identifiability analysis (the commented block at the end
    of identification.m): condition number and parameter correlations."""
    J = onp.asarray(J)
    s = onp.linalg.svd(J, compute_uv=False)
    out = {"singular_values": s,
           "condition_number": float(s[0] / s[-1]) if s[-1] > 0 else onp.inf}
    JtJ = J.T @ J
    try:
        C = onp.linalg.inv(JtJ)
        d = onp.sqrt(onp.diag(C))
        out["param_correlation"] = C / onp.outer(d, d)
    except onp.linalg.LinAlgError:
        out["param_correlation"] = None
    return out


def generate_time_window_weights(time, t_start, t_end, *, steepness=10.0,
                                 base_weight=0.1, peak_weight=1.0):
    """Port of the smooth sigmoid time-window weighting in identification.m."""
    time = onp.asarray(time, dtype=float).ravel()
    rise = 1.0 / (1.0 + onp.exp(-steepness * (time - t_start)))
    fall = 1.0 / (1.0 + onp.exp(-steepness * (time - t_end)))
    window = rise - fall
    w = base_weight + (peak_weight - base_weight) * window
    return onp.maximum(w, 0.0)