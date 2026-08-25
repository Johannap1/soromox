import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from tools.benchmarks._benchmark_common import (
    build_system_with_gauss_points,
    get_gvs_basis_order_system_config,
    get_system_registry,
)

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("system_name", list(get_system_registry()))
def test_benchmark_registry_builds_current_system_api(system_name: str) -> None:
    config = get_system_registry()[system_name]

    system = build_system_with_gauss_points(config, size=1, gauss_points=None)
    context = config.build_context(system)

    assert context["q"].shape == context["qd"].shape
    assert context["u"].shape == (system.num_actuators,)
    assert context["tau_ext"].shape == context["q"].shape
    assert context["y"].shape == (2 * context["q"].shape[0],)


def test_gvs_basis_order_registry_builds_current_system_api() -> None:
    config = get_gvs_basis_order_system_config()

    system = build_system_with_gauss_points(config, size=0, gauss_points=5)
    context = config.build_context(system)

    assert context["q"].shape == (system.num_dofs,)
    assert context["u"].shape == (system.num_actuators,)


@pytest.mark.parametrize("system_name", list(get_system_registry()))
def test_benchmark_systems_support_optional_coriolis_dynamics(
    system_name: str,
) -> None:
    config = get_system_registry()[system_name]
    default = build_system_with_gauss_points(config, size=1, gauss_points=None)
    enabled = build_system_with_gauss_points(
        config, size=1, gauss_points=None, consider_coriolis=True
    )
    disabled = build_system_with_gauss_points(
        config, size=1, gauss_points=None, consider_coriolis=False
    )
    context = config.build_context(enabled)
    q, qd = context["q"], context["qd"]

    M_enabled, Cqd_enabled, G_enabled = enabled.dynamics_terms(q, qd)
    M_default, Cqd_default, G_default = default.dynamics_terms(q, qd)
    M_disabled, Cqd_disabled, G_disabled = disabled.dynamics_terms(q, qd)

    assert default.consider_coriolis is True
    assert enabled.consider_coriolis is True
    assert disabled.consider_coriolis is False
    assert_allclose(M_default, M_enabled, rtol=1e-9, atol=1e-11)
    assert_allclose(Cqd_default, Cqd_enabled, rtol=1e-9, atol=1e-11)
    assert_allclose(G_default, G_enabled, rtol=1e-9, atol=1e-11)
    assert_allclose(M_disabled, M_enabled, rtol=1e-9, atol=1e-11)
    assert_allclose(G_disabled, G_enabled, rtol=1e-9, atol=1e-11)
    assert_allclose(Cqd_disabled, jnp.zeros_like(qd), rtol=0.0, atol=0.0)
    assert jnp.all(jnp.isfinite(Cqd_enabled))
    assert_allclose(
        disabled.coriolis_matrix(q, qd),
        enabled.coriolis_matrix(q, qd),
        rtol=1e-9,
        atol=1e-11,
    )

    tau_u = disabled.actuation_force(q, context["u"], qd=qd)
    rhs = (
        tau_u
        + context["tau_ext"]
        - G_disabled
        - disabled.elastic_force(q)
        - disabled.damping_matrix(q) @ qd
    )
    expected_qdd = jnp.linalg.solve(M_disabled, rhs)
    actual = disabled.forward_dynamics(
        context["t"],
        context["y"],
        (context["u"], context["tau_ext"]),
    )
    assert_allclose(actual, jnp.concatenate([qd, expected_qdd]), rtol=1e-8, atol=1e-10)
    assert_allclose(
        default.forward_dynamics(
            context["t"],
            context["y"],
            (context["u"], context["tau_ext"]),
        ),
        enabled.forward_dynamics(
            context["t"],
            context["y"],
            (context["u"], context["tau_ext"]),
        ),
        rtol=1e-9,
        atol=1e-11,
    )

    if hasattr(disabled, "with_params"):
        assert disabled.with_params(disabled.params).consider_coriolis is False


@pytest.mark.parametrize(
    ("system_name", "guarded_helpers", "exercise_forward_dynamics"),
    [
        ("pendulum", ("coriolis_matrix",), True),
        (
            "articulated_soft_robot",
            ("_motion_cross", "_force_cross"),
            True,
        ),
        ("planar_pcs", ("_dynamics_integration_kinematics",), False),
        ("pcs", ("_dynamics_integration_kinematics",), False),
        ("gvs", ("_joint_jacobian_time_derivative_step_terms",), False),
    ],
)
def test_disabled_paths_do_not_evaluate_derivative_or_convective_helpers(
    monkeypatch: pytest.MonkeyPatch,
    system_name: str,
    guarded_helpers: tuple[str, ...],
    exercise_forward_dynamics: bool,
) -> None:
    config = get_system_registry()[system_name]
    system = build_system_with_gauss_points(
        config, size=1, gauss_points=None, consider_coriolis=False
    )
    context = config.build_context(system)

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("disabled dynamics evaluated a guarded helper")

    for helper_name in guarded_helpers:
        monkeypatch.setattr(type(system), helper_name, fail_if_called)

    with jax.disable_jit():
        if exercise_forward_dynamics:
            system.forward_dynamics(
                context["t"],
                context["y"],
                (context["u"], context["tau_ext"]),
            )
        else:
            system.dynamics_terms(context["q"], context["qd"])
