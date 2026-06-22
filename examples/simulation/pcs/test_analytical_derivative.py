import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from soromox.systems import PCS, PCSParams

jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)


if __name__ == "__main__":
    num_segments = 2
    seed = 7212

    rho = 1070 * jnp.ones((num_segments,))
    segment_lengths = 1e-1 * jnp.ones((num_segments,))
    damping_matrix = 1e-3 * jnp.diag(
        (
            jnp.repeat(
                jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0
            )
            * segment_lengths[:, None]
        ).flatten()
    )
    params = PCSParams(
        base_pose=jnp.array([jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0]),
        length=segment_lengths,
        radius=2e-2 * jnp.ones((num_segments,)),
        density=rho,
        gravity=jnp.array([0.0, 0.0, 9.81]),
        young_modulus=2e3 * jnp.ones((num_segments,)),
        shear_modulus=1e3 * jnp.ones((num_segments,)),
        damping_matrix=damping_matrix,
        reference_strain=jnp.tile(
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), num_segments
        ),
    )

    robot = PCS(params=params)

    key_q, key_qd, key_u, key_tau = jax.random.split(jax.random.PRNGKey(seed), 4)
    q = 0.5 * jax.random.normal(key_q, (robot.num_dofs,))
    qd = 0.2 * jax.random.normal(key_qd, (robot.num_dofs,))
    u = 1e-2 * jax.random.normal(key_u, (robot.num_actuators,))
    tau_ext = 1e-2 * jax.random.normal(key_tau, (robot.num_dofs,))
    y = jnp.concatenate([q, qd])
    t = jnp.array(0.0)

    yd = robot.forward_dynamics(t, y, (u, tau_ext))
    _, qdd = jnp.split(yd, 2)

    analytical_dID_dq, analytical_dID_dqd = robot.inverse_dynamics_derivatives(
        q, qd, qdd
    )
    analytical_dtau_el_dq = robot.elastic_force_derivative_q(q)
    analytical_dtau_damp_dq, analytical_dtau_damp_dqd = robot.damping_force_derivatives(
        q, qd
    )
    analytical_dtau_u_dq = robot.actuation_force_derivative_q(q, u)
    analytical_dtau_u_du = robot.actuation_force_derivative_u(q)
    analytical_dqdd_dq, analytical_dqdd_dqd = robot.forward_dynamics_derivatives(
        q, qd, qdd, u
    )
    analytical_dqdd_du, analytical_dqdd_dtau_ext = (
        robot.forward_dynamics_input_derivatives(q)
    )
    analytical_dyd_dy = robot.forward_dynamics_state_jacobian(t, y, (u, tau_ext))
    analytical_dyd_dy_full, analytical_dyd_du, analytical_dyd_dtau_ext = (
        robot.forward_dynamics_jacobians(t, y, (u, tau_ext))
    )

    autodiff_dID_dq = jax.jacfwd(
        lambda q_arg: robot.inverse_dynamics_force(q_arg, qd, qdd)
    )(q)
    autodiff_dID_dqd = jax.jacfwd(
        lambda qd_arg: robot.inverse_dynamics_force(q, qd_arg, qdd)
    )(qd)
    autodiff_dtau_el_dq = jax.jacfwd(lambda q_arg: robot.elastic_force(q_arg))(q)
    autodiff_dtau_damp_dq = jax.jacfwd(lambda q_arg: robot.damping_matrix(q_arg) @ qd)(
        q
    )
    autodiff_dtau_damp_dqd = jax.jacfwd(
        lambda qd_arg: robot.damping_matrix(q) @ qd_arg
    )(qd)
    autodiff_dtau_u_dq = jax.jacfwd(lambda q_arg: robot.actuation_force(q_arg, u))(q)
    autodiff_dtau_u_du = jax.jacfwd(lambda u_arg: robot.actuation_force(q, u_arg))(u)
    autodiff_dqdd_dq = jax.jacfwd(
        lambda q_arg: jnp.split(
            robot.forward_dynamics(t, jnp.concatenate([q_arg, qd]), (u, tau_ext)),
            2,
        )[1]
    )(q)
    autodiff_dqdd_dqd = jax.jacfwd(
        lambda qd_arg: jnp.split(
            robot.forward_dynamics(t, jnp.concatenate([q, qd_arg]), (u, tau_ext)),
            2,
        )[1]
    )(qd)
    autodiff_dqdd_du = jax.jacfwd(
        lambda u_arg: jnp.split(robot.forward_dynamics(t, y, (u_arg, tau_ext)), 2)[1]
    )(u)
    autodiff_dqdd_dtau_ext = jax.jacfwd(
        lambda tau_ext_arg: jnp.split(
            robot.forward_dynamics(t, y, (u, tau_ext_arg)),
            2,
        )[1]
    )(tau_ext)
    autodiff_dyd_dy = jax.jacfwd(
        lambda y_arg: robot.forward_dynamics(t, y_arg, (u, tau_ext))
    )(y)
    autodiff_dyd_du = jax.jacfwd(
        lambda u_arg: robot.forward_dynamics(t, y, (u_arg, tau_ext))
    )(u)
    autodiff_dyd_dtau_ext = jax.jacfwd(
        lambda tau_ext_arg: robot.forward_dynamics(t, y, (u, tau_ext_arg))
    )(tau_ext)

    comparisons = [
        ("inverse_dynamics_derivatives dID/dq", analytical_dID_dq, autodiff_dID_dq),
        ("inverse_dynamics_derivatives dID/dqd", analytical_dID_dqd, autodiff_dID_dqd),
        ("elastic_force_derivative_q", analytical_dtau_el_dq, autodiff_dtau_el_dq),
        (
            "damping_force_derivatives dtau_damp/dq",
            analytical_dtau_damp_dq,
            autodiff_dtau_damp_dq,
        ),
        (
            "damping_force_derivatives dtau_damp/dqd",
            analytical_dtau_damp_dqd,
            autodiff_dtau_damp_dqd,
        ),
        (
            "actuation_force_derivative_q",
            analytical_dtau_u_dq,
            autodiff_dtau_u_dq,
        ),
        (
            "actuation_force_derivative_u",
            analytical_dtau_u_du,
            autodiff_dtau_u_du,
        ),
        ("forward_dynamics_derivatives dqdd/dq", analytical_dqdd_dq, autodiff_dqdd_dq),
        (
            "forward_dynamics_derivatives dqdd/dqd",
            analytical_dqdd_dqd,
            autodiff_dqdd_dqd,
        ),
        (
            "forward_dynamics_input_derivatives dqdd/du",
            analytical_dqdd_du,
            autodiff_dqdd_du,
        ),
        (
            "forward_dynamics_input_derivatives dqdd/dtau_ext",
            analytical_dqdd_dtau_ext,
            autodiff_dqdd_dtau_ext,
        ),
        ("forward_dynamics_state_jacobian dyd/dy", analytical_dyd_dy, autodiff_dyd_dy),
        (
            "forward_dynamics_jacobians dyd/dy",
            analytical_dyd_dy_full,
            autodiff_dyd_dy,
        ),
        ("forward_dynamics_jacobians dyd/du", analytical_dyd_du, autodiff_dyd_du),
        (
            "forward_dynamics_jacobians dyd/dtau_ext",
            analytical_dyd_dtau_ext,
            autodiff_dyd_dtau_ext,
        ),
    ]

    for name, analytical, autodiff in comparisons:
        max_abs_error = jnp.max(jnp.abs(analytical - autodiff))
        rel_error = max_abs_error / jnp.maximum(1.0, jnp.max(jnp.abs(autodiff)))
        print(
            f"{name}: rel error = {float(rel_error):.6e}, "
            f"max abs error = {float(max_abs_error):.6e}"
        )
