from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import AbstractSolver, Euler, Tsit5
from numpy.testing import assert_allclose
from system_param_builders import pcs_params, pendulum_params

from soromox.simulation import IMEXEuler, IMEXEulerImplicitContact, SemiImplicitEuler
from soromox.systems import (
    PCS,
    PCSStructure,
    Pendulum,
    PlanarHSA,
    PlanarHSAParams,
    PlanarHSAStructure,
    SystemState,
)
from soromox.systems.dynamical_system import _RolloutState


def _pendulum() -> Pendulum:
    return Pendulum(
        pendulum_params(
            length=jnp.array([0.5]),
            center_of_mass_length=jnp.array([0.25]),
            mass=jnp.array([1.0]),
            moment_inertia=jnp.array([0.1]),
            gravity=jnp.array([0.0, -9.81]),
            joint_stiffness=jnp.array([[2.0]]),
            joint_damping=jnp.array([[0.1]]),
        )
    )


def _fixed_spatial_pcs() -> PCS:
    return PCS(
        params=pcs_params(
            length=jnp.array([0.1]),
            radius=jnp.array([0.01]),
            density=jnp.array([1000.0]),
            young_modulus=jnp.array([1.0e5]),
            shear_modulus=jnp.array([4.0e4]),
            damping_matrix=jnp.eye(6),
            gravity=jnp.array([0.0, 0.0, -9.81]),
        ),
        structure=PCSStructure(num_gauss_points=3),
        floating_base=False,
        backend="jax",
    )


def _floating_spatial_pcs() -> PCS:
    return PCS(
        params=pcs_params(
            length=jnp.array([0.1]),
            radius=jnp.array([0.01]),
            density=jnp.array([1000.0]),
            young_modulus=jnp.array([1.0e5]),
            shear_modulus=jnp.array([4.0e4]),
            damping_matrix=jnp.eye(6),
            gravity=jnp.array([0.0, 0.0, -9.81]),
        ),
        structure=PCSStructure(num_gauss_points=3),
        floating_base=True,
        backend="jax",
    )


def _floating_hysteretic_planar_hsa() -> PlanarHSA:
    params_path = (
        Path(__file__).resolve().parents[2]
        / "assets"
        / "robot_parameters"
        / "planar_hsa"
        / "fpu_control.npz"
    )
    params = PlanarHSAParams.from_npz(params_path)
    num_strains = 3 * params.length.shape[0]
    params = params.replace(
        hysteresis_basis=jnp.ones((num_strains, 1)),
        hysteresis_alpha=0.5 * jnp.ones((num_strains,)),
        hysteresis_A=jnp.ones((1,)),
        hysteresis_n=jnp.ones((1,)),
        hysteresis_beta=0.5 * jnp.ones((1,)),
        hysteresis_gamma=0.5 * jnp.ones((1,)),
    )
    return PlanarHSA(
        params=params,
        structure=PlanarHSAStructure(consider_hysteresis=True),
        floating_base=True,
    )


def test_semi_implicit_euler_is_a_deterministic_diffrax_solver():
    solver = SemiImplicitEuler(_pendulum())

    assert isinstance(solver, AbstractSolver)
    assert not isinstance(solver, Euler)


def test_semi_implicit_euler_applies_kick_before_drift():
    robot = _pendulum()
    q0 = jnp.array([0.2])
    qd0 = jnp.array([0.3])
    u = jnp.zeros((robot.num_actuators,))
    tau_ext = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, qd0])
    dt = 1e-4

    derivative = robot.forward_dynamics(jnp.asarray(0.0), y0, (u, tau_ext))
    _, qdd0 = jnp.split(derivative, 2)
    qd1 = qd0 + dt * qdd0
    expected = jnp.concatenate([q0 + dt * qd1, qd1])

    trajectory = robot.rollout_to(
        initial_state=SystemState(t=0.0, y=y0),
        u=u,
        tau_ext=tau_ext,
        t1=dt,
        solver_dt=dt,
        save_ts=jnp.array([0.0, dt]),
        solver=SemiImplicitEuler(robot),
    )

    assert jnp.allclose(trajectory.y[-1], expected)


def test_semi_implicit_euler_rollout_is_jittable_and_vmappable():
    robot = _pendulum()
    solver = SemiImplicitEuler(robot)
    q0s = jnp.array([[0.1], [0.2], [-0.15]])
    qd0s = jnp.array([[0.0], [0.1], [-0.05]])

    def rollout(q0, qd0):
        trajectory = robot.rollout_to(
            initial_state=SystemState(t=0.0, y=jnp.concatenate([q0, qd0])),
            t1=0.002,
            solver_dt=1e-4,
            save_ts=jnp.array([0.0, 0.001, 0.002]),
            solver=solver,
        )
        return trajectory.y

    trajectories = jax.jit(jax.vmap(rollout))(q0s, qd0s)

    assert trajectories.shape == (3, 3, 2)
    assert jnp.all(jnp.isfinite(trajectories))


def test_semi_implicit_euler_preserves_absent_optional_rollout_states():
    robot = _pendulum()

    trajectory = robot.rollout_to(
        initial_state=SystemState(t=0.0, y=jnp.array([0.1, 0.0])),
        t1=1e-4,
        solver_dt=1e-4,
        save_ts=jnp.array([0.0, 1e-4]),
        solver=SemiImplicitEuler(robot),
    )

    assert trajectory.control_state is None
    assert trajectory.environment_state is None


def test_semi_implicit_euler_supports_spatial_floating_base_state():
    robot = _floating_spatial_pcs()
    q0 = robot.pack_configuration(
        jnp.zeros((robot.num_internal_dofs,)),
        base_pose=jnp.array([0.1, -0.2, 0.3, 0.9, 0.2, -0.1, 0.4]),
    )
    qd0 = robot.pack_velocity(
        jnp.zeros((robot.num_internal_dofs,)),
        base_velocity=jnp.array([0.01, -0.02, 0.005, 0.03, -0.01, 0.02]),
    )
    initial_y = robot.pack_state(q0, qd0)
    dt = 1e-5

    solver_y0 = robot.project_state(initial_y)
    solver_q0, solver_qd0, _ = robot.split_state(solver_y0)
    dynamics_y0 = robot.project_state(solver_y0)
    derivative = robot.forward_dynamics(jnp.asarray(0.0), dynamics_y0)
    _, qdd0, _ = robot.split_state(derivative)
    qd1 = solver_qd0 + dt * qdd0
    expected_q1 = robot.retract_configuration(solver_q0, dt * qd1)

    trajectory = robot.rollout_to(
        initial_state=SystemState(t=0.0, y=initial_y),
        t1=dt,
        solver_dt=dt,
        save_ts=jnp.array([0.0, dt]),
        solver=SemiImplicitEuler(robot),
    )
    q1, actual_qd1, auxiliary1 = robot.split_state(trajectory.y[-1])

    assert robot.num_coordinates != robot.num_velocities
    assert_allclose(q1, expected_q1, atol=1e-18)
    assert_allclose(actual_qd1, qd1, atol=1e-15)
    assert auxiliary1.shape == (0,)
    assert_allclose(jnp.linalg.norm(q1[:4]), 1.0)


def test_semi_implicit_euler_advances_system_auxiliary_state():
    robot = _floating_hysteretic_planar_hsa()
    q0 = robot.pack_configuration(
        jnp.zeros((robot.num_internal_dofs,)),
        base_pose=jnp.zeros((robot.num_base_coordinates,)),
    )
    qd0 = robot.pack_velocity(
        1e-3 * jnp.ones((robot.num_internal_dofs,)),
        base_velocity=jnp.zeros((robot.num_base_velocities,)),
    )
    auxiliary0 = 0.1 * jnp.ones((robot.num_auxiliary_states,))
    y0 = robot.pack_state(q0, qd0, auxiliary0)
    dt = 1e-7

    derivative = robot.forward_dynamics(jnp.asarray(0.0), y0)
    _, qdd0, auxiliary_derivative0 = robot.split_state(derivative)
    expected_qd1 = qd0 + dt * qdd0
    expected_q1 = robot.retract_configuration(q0, dt * expected_qd1)
    expected_auxiliary1 = auxiliary0 + dt * auxiliary_derivative0

    trajectory = robot.rollout_to(
        initial_state=SystemState(t=0.0, y=y0),
        t1=dt,
        solver_dt=dt,
        save_ts=jnp.array([0.0, dt]),
        solver=SemiImplicitEuler(robot),
    )
    q1, qd1, auxiliary1 = robot.split_state(trajectory.y[-1])

    assert robot.num_auxiliary_states == 1
    assert_allclose(q1, expected_q1)
    assert_allclose(qd1, expected_qd1, atol=1e-9)
    assert_allclose(auxiliary1, expected_auxiliary1)


def test_tsit5_supports_spatial_floating_base_state():
    robot = _floating_spatial_pcs()
    q0 = robot.pack_configuration(
        jnp.zeros((robot.num_internal_dofs,)),
        base_pose=jnp.array([0.1, -0.2, 0.3, 0.9, 0.2, -0.1, 0.4]),
    )
    qd0 = robot.pack_velocity(
        jnp.zeros((robot.num_internal_dofs,)),
        base_velocity=jnp.array([0.01, -0.02, 0.005, 0.03, -0.01, 0.02]),
    )

    dt = 1e-7
    trajectory = robot.rollout_to(
        initial_state=SystemState(t=0.0, y=robot.pack_state(q0, qd0)),
        t1=dt,
        solver_dt=dt,
        save_ts=jnp.array([0.0, dt]),
        solver=Tsit5(),
    )
    q, qd, auxiliary = robot.split_state(trajectory.y)

    assert trajectory.y.shape[-1] == robot.state_size
    assert q.shape[-1] == robot.num_coordinates
    assert qd.shape[-1] == robot.num_velocities
    assert auxiliary.shape[-1] == robot.num_auxiliary_states
    assert jnp.all(jnp.isfinite(trajectory.y))
    assert_allclose(jnp.linalg.norm(q[..., :4], axis=-1), 1.0)


def test_imex_euler_is_a_deterministic_diffrax_solver():
    solver = IMEXEuler(_pendulum())

    assert isinstance(solver, AbstractSolver)
    assert not isinstance(solver, Euler)


def test_imex_euler_matches_hand_derived_step():
    robot = _pendulum()
    q0 = jnp.array([0.2])
    qd0 = jnp.array([0.3])
    u = jnp.zeros((robot.num_actuators,))
    tau_ext = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, qd0])
    dt = 1e-4

    M, Cqd, G = robot.dynamics_terms(q0, qd0)
    K = robot.stiffness_matrix()
    D = robot.damping_matrix(q0)
    tau_u = robot.actuation_force(q0, u, qd=qd0)
    elastic_nonlinear = robot.elastic_force(q0) - K @ q0
    rhs = tau_u + tau_ext - Cqd - G - elastic_nonlinear
    A = M + dt * D + dt**2 * K
    b = M @ qd0 + dt * (rhs - K @ q0)
    qd1 = jnp.linalg.solve(A, b)
    expected = jnp.concatenate([q0 + dt * qd1, qd1])

    trajectory = robot.rollout_to(
        initial_state=SystemState(t=0.0, y=y0),
        u=u,
        tau_ext=tau_ext,
        t1=dt,
        solver_dt=dt,
        save_ts=jnp.array([0.0, dt]),
        solver=IMEXEuler(robot),
    )

    assert_allclose(trajectory.y[-1], expected)


def test_imex_euler_rollout_is_jittable_and_vmappable():
    robot = _pendulum()
    solver = IMEXEuler(robot)
    q0s = jnp.array([[0.1], [0.2], [-0.15]])
    qd0s = jnp.array([[0.0], [0.1], [-0.05]])

    def rollout(q0, qd0):
        trajectory = robot.rollout_to(
            initial_state=SystemState(t=0.0, y=jnp.concatenate([q0, qd0])),
            t1=0.002,
            solver_dt=1e-4,
            save_ts=jnp.array([0.0, 0.001, 0.002]),
            solver=solver,
        )
        return trajectory.y

    trajectories = jax.jit(jax.vmap(rollout))(q0s, qd0s)

    assert trajectories.shape == (3, 3, 2)
    assert jnp.all(jnp.isfinite(trajectories))


def test_imex_euler_preserves_absent_optional_rollout_states():
    robot = _pendulum()

    trajectory = robot.rollout_to(
        initial_state=SystemState(t=0.0, y=jnp.array([0.1, 0.0])),
        t1=1e-4,
        solver_dt=1e-4,
        save_ts=jnp.array([0.0, 1e-4]),
        solver=IMEXEuler(robot),
    )

    assert trajectory.control_state is None
    assert trajectory.environment_state is None


def test_imex_euler_supports_spatial_floating_base_state():
    robot = _floating_spatial_pcs()
    q0 = robot.pack_configuration(
        jnp.zeros((robot.num_internal_dofs,)),
        base_pose=jnp.array([0.1, -0.2, 0.3, 0.9, 0.2, -0.1, 0.4]),
    )
    qd0 = robot.pack_velocity(
        jnp.zeros((robot.num_internal_dofs,)),
        base_velocity=jnp.array([0.01, -0.02, 0.005, 0.03, -0.01, 0.02]),
    )
    initial_y = robot.pack_state(q0, qd0)
    dt = 1e-7

    u = jnp.zeros((robot.num_actuators,))
    tau_ext = jnp.zeros_like(qd0)
    _, q_internal = robot.split_configuration(q0)
    q_embedded = jnp.concatenate(
        [jnp.zeros((robot.num_base_velocities,)), q_internal]
    )
    M, Cqd, G = robot.dynamics_terms(q0, qd0)
    K = robot.stiffness_matrix()
    D = robot.damping_matrix(q0)
    tau_u = robot.actuation_force(q0, u, qd=qd0)
    elastic_nonlinear = robot.elastic_force(q0) - K @ q_embedded
    rhs = tau_u + tau_ext - Cqd - G - elastic_nonlinear
    A = M + dt * D + dt**2 * K
    b = M @ qd0 + dt * (rhs - K @ q_embedded)
    expected_qd1 = jnp.linalg.solve(A, b)
    expected_q1 = robot.retract_configuration(q0, dt * expected_qd1)

    trajectory = robot.rollout_to(
        initial_state=SystemState(t=0.0, y=initial_y),
        t1=dt,
        solver_dt=dt,
        save_ts=jnp.array([0.0, dt]),
        solver=IMEXEuler(robot),
    )
    q1, actual_qd1, auxiliary1 = robot.split_state(trajectory.y[-1])

    assert robot.num_coordinates != robot.num_velocities
    assert_allclose(q1, expected_q1, atol=1e-18)
    assert_allclose(actual_qd1, expected_qd1, atol=1e-15)
    assert auxiliary1.shape == (0,)
    assert_allclose(jnp.linalg.norm(q1[:4]), 1.0, atol=1e-6)


class _ZeroController(eqx.Module):
    num_actuators: int

    def __call__(self, state: SystemState):
        return jnp.zeros((self.num_actuators,)), None


class _IntegralController(eqx.Module):
    kp: float
    ki: float
    target: jnp.ndarray

    def __call__(self, state: SystemState):
        q = state.y[: self.target.shape[0]]
        error = self.target - q
        u = self.kp * error + self.ki * state.control_state
        control_state_dot = error
        return u, control_state_dot


def test_imex_euler_matches_hand_derived_closed_loop_step():
    robot = _pendulum()
    q0 = jnp.array([0.2])
    qd0 = jnp.array([0.3])
    control_state0 = jnp.array([0.05])
    controller = _IntegralController(kp=1.5, ki=0.5, target=jnp.array([0.0]))
    tau_ext = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, qd0])
    dt = 1e-4

    error = controller.target - q0
    u = controller.kp * error + controller.ki * control_state0
    expected_control_state1 = control_state0 + dt * error

    M, Cqd, G = robot.dynamics_terms(q0, qd0)
    K = robot.stiffness_matrix()
    D = robot.damping_matrix(q0)
    tau_u = robot.actuation_force(q0, u, qd=qd0)
    elastic_nonlinear = robot.elastic_force(q0) - K @ q0
    rhs = tau_u + tau_ext - Cqd - G - elastic_nonlinear
    A = M + dt * D + dt**2 * K
    b = M @ qd0 + dt * (rhs - K @ q0)
    qd1 = jnp.linalg.solve(A, b)
    expected = jnp.concatenate([q0 + dt * qd1, qd1])

    trajectory = robot.rollout_closed_loop_to(
        initial_state=SystemState(t=0.0, y=y0, control_state=control_state0),
        controller=controller,
        tau_ext=tau_ext,
        t1=dt,
        solver_dt=dt,
        save_ts=jnp.array([0.0, dt]),
        solver=IMEXEuler(robot),
    )

    assert_allclose(trajectory.y[-1], expected)
    assert_allclose(trajectory.control_state[-1], expected_control_state1)


def test_imex_euler_closed_loop_rollout_is_jittable_and_vmappable():
    robot = _pendulum()
    solver = IMEXEuler(robot)
    controller = _ZeroController(num_actuators=robot.num_actuators)
    q0s = jnp.array([[0.1], [0.2], [-0.15]])
    qd0s = jnp.array([[0.0], [0.1], [-0.05]])

    def rollout(q0, qd0):
        trajectory = robot.rollout_closed_loop_to(
            initial_state=SystemState(t=0.0, y=jnp.concatenate([q0, qd0])),
            controller=controller,
            t1=0.002,
            solver_dt=1e-4,
            save_ts=jnp.array([0.0, 0.001, 0.002]),
            solver=solver,
        )
        return trajectory.y

    trajectories = jax.jit(jax.vmap(rollout))(q0s, qd0s)

    assert trajectories.shape == (3, 3, 2)
    assert jnp.all(jnp.isfinite(trajectories))


def test_imex_euler_implicit_contact_reduces_to_plain_imex_when_not_touching():
    """With the contact body placed far away, K_contact and tau_contact are
    both identically zero, so the rod update should exactly match plain
    IMEXEuler's. Uses a fixed-base fixture: the current implementation
    assumes num_coordinates == num_velocities, which floating-base systems
    violate (their configuration carries a redundant quaternion component
    with no matching velocity component) -- see the
    IMEXEulerImplicitContact docstring."""
    robot = _fixed_spatial_pcs()
    q0 = jnp.zeros((robot.num_internal_dofs,))
    qd0 = jnp.array([0.01, -0.02, 0.005, 0.03, -0.01, 0.02])[: robot.num_internal_dofs]
    y0_state = jnp.concatenate([q0, qd0])
    dt = 1e-4
    u = jnp.zeros((robot.num_actuators,))

    plain_trajectory = robot.rollout_to(
        initial_state=SystemState(t=0.0, y=y0_state),
        u=u,
        t1=dt,
        solver_dt=dt,
        save_ts=jnp.array([0.0, dt]),
        solver=IMEXEuler(robot),
    )

    solver = IMEXEulerImplicitContact(robot)
    y0 = _RolloutState(y=y0_state, environment_state=(-1000.0, 0.0))
    y1, *_ = solver.step(
        terms=None,
        t0=jnp.asarray(0.0),
        t1=jnp.asarray(dt),
        y0=y0,
        args=(u, 1.0, 0.0, 2500.0),
        solver_state=None,
        made_jump=None,
    )

    # atol relaxed vs. other tests in this file: those compare against a
    # hand-derived computation using the same operation order (so float32
    # rounding cancels identically); this compares across two genuinely
    # different code paths (rollout_to's Diffrax wrapper vs. calling
    # .step() directly), which don't round identically even when
    # mathematically equivalent.
    assert_allclose(y1.y, plain_trajectory.y[-1], atol=1e-6)


def test_imex_euler_implicit_contact_survives_large_dt_with_light_body():
    """Deep, sustained contact against a light body at a dt/stiffness
    combination -- should stay finite throughout, unlike plain explicit
    treatment of the same contact term."""
    robot = _fixed_spatial_pcs()
    q0 = jnp.zeros((robot.num_internal_dofs,))
    qd0 = jnp.zeros((robot.num_internal_dofs,))
    y0_state = jnp.concatenate([q0, qd0])

    initial_tip_y = robot.forward_kinematics_tips(q0)[-1][1, 3]
    body_y0 = initial_tip_y + 0.01  # already overlapping at t=0

    solver = IMEXEulerImplicitContact(robot)
    dt = 1e-2
    stiffness = 2500.0
    body_mass = 0.05
    body_damping = 0.5
    u = jnp.zeros((robot.num_actuators,))

    y0 = _RolloutState(y=y0_state, environment_state=(body_y0, 0.0))
    for step in range(20):
        t0 = step * dt
        y1, *_ = solver.step(
            terms=None,
            t0=jnp.asarray(t0),
            t1=jnp.asarray(t0 + dt),
            y0=y0,
            args=(u, body_mass, body_damping, stiffness),
            solver_state=None,
            made_jump=None,
        )
        assert jnp.all(jnp.isfinite(y1.y))
        body_y1, body_vy1 = y1.environment_state
        assert jnp.isfinite(body_y1)
        assert jnp.isfinite(body_vy1)
        y0 = y1


def test_imex_euler_advances_system_auxiliary_state():
    robot = _floating_hysteretic_planar_hsa()
    q0 = robot.pack_configuration(
        jnp.zeros((robot.num_internal_dofs,)),
        base_pose=jnp.zeros((robot.num_base_coordinates,)),
    )
    qd0 = robot.pack_velocity(
        1e-3 * jnp.ones((robot.num_internal_dofs,)),
        base_velocity=jnp.zeros((robot.num_base_velocities,)),
    )
    auxiliary0 = 0.1 * jnp.ones((robot.num_auxiliary_states,))
    y0 = robot.pack_state(q0, qd0, auxiliary0)
    dt = 1e-7

    derivative = robot.forward_dynamics(jnp.asarray(0.0), y0)
    _, _, auxiliary_derivative0 = robot.split_state(derivative)
    expected_auxiliary1 = auxiliary0 + dt * auxiliary_derivative0

    trajectory = robot.rollout_to(
        initial_state=SystemState(t=0.0, y=y0),
        t1=dt,
        solver_dt=dt,
        save_ts=jnp.array([0.0, dt]),
        solver=IMEXEuler(robot),
    )
    _, _, auxiliary1 = robot.split_state(trajectory.y[-1])

    assert robot.num_auxiliary_states == 1
    assert_allclose(auxiliary1, expected_auxiliary1)
