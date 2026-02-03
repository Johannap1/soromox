import equinox as eqx
from jax import value_and_grad
from jax import numpy as jnp

from soromox.systems import Pendulum, SystemState


def _pendulum_params():
    # Gravity set to zero so the zero configuration is an equilibrium for zero actuation.
    return {
        "L": jnp.array([0.5, 0.3]),
        "Lc": jnp.array([0.25, 0.15]),
        "m": jnp.array([1.0, 0.5]),
        "I": jnp.array([0.1, 0.05]),
        "g": jnp.array([0.0, 0.0]),
    }


class PIDController(eqx.Module):
    kp: float
    ki: float
    kd: float
    target: jnp.ndarray

    def __call__(self, state: SystemState):
        q_len = self.target.shape[0]
        q, qd = jnp.split(state.y, [q_len], axis=0)
        integ = state.control_state

        error = self.target - q
        d_error = -qd  # derivative of (target - q) when target is constant

        u = self.kp * error + self.ki * integ + self.kd * d_error
        control_state_dot = error
        return u, control_state_dot


def test_rollout_to_keeps_equilibrium():
    robot = Pendulum(_pendulum_params())

    q0 = jnp.zeros((2,))
    qd0 = jnp.zeros_like(q0)
    initial_state = SystemState(t=0.0, y=jnp.concatenate([q0, qd0]))

    trajectory = robot.rollout_to(
        initial_state=initial_state,
        u=jnp.zeros_like(q0),
        t1=0.1,
        solver_dt=1e-3,
        save_dt=0.01,
    )

    assert trajectory.control_state is None
    assert jnp.allclose(trajectory.y, 0.0, atol=1e-6)


def test_rollout_closed_loop_to_tracks_target():
    robot = Pendulum(_pendulum_params())

    q0 = jnp.array([0.2, -0.1])
    qd0 = jnp.zeros_like(q0)
    initial_state = SystemState(
        t=0.0,
        y=jnp.concatenate([q0, qd0]),
        u=jnp.zeros_like(q0),
        control_state=jnp.zeros_like(q0),  # integral term
    )

    controller = PIDController(kp=10.0, ki=3.0, kd=2.0, target=jnp.zeros_like(q0))

    trajectory = robot.rollout_closed_loop_to(
        initial_state=initial_state,
        controller=controller,
        t1=0.4,
        solver_dt=1e-3,
        save_dt=0.02,
    )

    q_final = trajectory.y[-1, : q0.shape[0]]
    assert jnp.linalg.norm(q_final) < 5e-2
    assert trajectory.control_state is not None


def test_rollout_discrete_closed_loop_to_tracks_target():
    robot = Pendulum(_pendulum_params())

    q0 = jnp.array([0.3, -0.2])
    qd0 = jnp.zeros_like(q0)
    initial_state = SystemState(
        t=0.0,
        y=jnp.concatenate([q0, qd0]),
        u=jnp.zeros_like(q0),
        control_state=jnp.zeros_like(q0),
    )

    controller = PIDController(kp=8.0, ki=2.5, kd=1.5, target=jnp.zeros_like(q0))

    trajectory = robot.rollout_discrete_closed_loop_to(
        initial_state=initial_state,
        controller=controller,
        t1=0.6,
        solver_dt=1e-3,
        control_dt=0.02,
        save_dt=0.02,
    )

    q_final = trajectory.y[-1, : q0.shape[0]]
    assert jnp.linalg.norm(q_final) < 7e-2
    assert trajectory.control_state is not None


def test_gradient_through_rollout_to():
    """Test that gradients can be computed through rollout_to (open loop).

    This test verifies that the fix to _compute_save_times (clipping save times)
    works correctly for open-loop rollouts.
    """
    robot = Pendulum(_pendulum_params())

    q0 = jnp.array([0.3, -0.2])
    qd0 = jnp.zeros_like(q0)

    # Simulation parameters
    t1 = 0.6
    solver_dt = 1e-3
    save_dt = 0.02
    max_steps = int((t1 / solver_dt) * 2)

    def loss_fn(u):
        initial_state = SystemState(
            t=0.0,
            y=jnp.concatenate([q0, qd0]),
        )

        trajectory = robot.rollout_to(
            initial_state=initial_state,
            u=u,
            t1=t1,
            solver_dt=solver_dt,
            save_dt=save_dt,
            max_steps=max_steps,
        )

        q_final = trajectory.y[-1, : q0.shape[0]]
        return jnp.sum(q_final**2)

    # Test gradient computation
    u = jnp.array([0.5, 0.5])
    loss_val, grad = value_and_grad(loss_fn)(u)

    # Verify gradient was computed (not NaN)
    assert jnp.isfinite(grad).all()


def test_gradient_through_rollout_closed_loop_to():
    """Test that gradients can be computed through rollout_closed_loop_to.

    This test verifies that the fix to _compute_save_times (clipping save times)
    works correctly for continuous closed-loop rollouts.
    """
    robot = Pendulum(_pendulum_params())

    q0 = jnp.array([0.3, -0.2])
    qd0 = jnp.zeros_like(q0)
    target = jnp.zeros_like(q0)

    # Simulation parameters
    t1 = 0.6
    solver_dt = 1e-3
    save_dt = 0.02
    max_steps = int((t1 / solver_dt) * 2)

    def loss_fn(kp):
        initial_state = SystemState(
            t=0.0,
            y=jnp.concatenate([q0, qd0]),
            u=jnp.zeros_like(q0),
            control_state=jnp.zeros_like(q0),
        )

        controller = PIDController(kp=kp, ki=2.5, kd=1.5, target=target)

        trajectory = robot.rollout_closed_loop_to(
            initial_state=initial_state,
            controller=controller,
            t1=t1,
            solver_dt=solver_dt,
            save_dt=save_dt,
            max_steps=max_steps,
        )

        q_final = trajectory.y[-1, : q0.shape[0]]
        return jnp.sum(q_final**2)

    # Test gradient computation
    kp = 8.0
    loss_val, grad = value_and_grad(loss_fn)(kp)

    # Verify gradient was computed (not NaN or zero)
    assert jnp.isfinite(grad)


def test_gradient_through_discrete_rollout():
    """Test that gradients can be computed through rollout_discrete_closed_loop_to.

    This test addresses the issue where computing gradients would fail with:
    _EquinoxRuntimeError: saveat.ts must lie between t0 and t1.

    The fix involved clipping save times in both _compute_save_times and in the
    dynamically constructed ts_control_interval to avoid floating-point precision
    issues during gradient computation.
    """
    robot = Pendulum(_pendulum_params())

    q0 = jnp.array([0.3, -0.2])
    qd0 = jnp.zeros_like(q0)
    target = jnp.zeros_like(q0)

    # Simulation parameters
    t1 = 0.6
    solver_dt = 1e-3
    control_dt = 0.02
    save_dt = 0.02
    # Calculate max_steps needed for gradient computation
    max_steps = int((t1 / solver_dt) * 2)  # 2x buffer

    def loss_fn(kp):
        initial_state = SystemState(
            t=0.0,
            y=jnp.concatenate([q0, qd0]),
            u=jnp.zeros_like(q0),
            control_state=jnp.zeros_like(q0),
        )

        controller = PIDController(kp=kp, ki=2.5, kd=1.5, target=target)

        trajectory = robot.rollout_discrete_closed_loop_to(
            initial_state=initial_state,
            controller=controller,
            t1=t1,
            solver_dt=solver_dt,
            control_dt=control_dt,
            save_dt=save_dt,
            max_steps=max_steps,  # Required for gradient computation
        )

        # Simple loss: final position error
        q_final = trajectory.y[-1, : q0.shape[0]]
        return jnp.sum(q_final**2)

    # Test gradient computation
    kp = 8.0
    loss_val, grad = value_and_grad(loss_fn)(kp)

    # Verify gradient was computed (not NaN or zero)
    assert jnp.isfinite(grad)
    assert grad != 0.0
