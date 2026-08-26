from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from tools.benchmarks import benchmark_gvs_jax_dynamics as benchmark


def test_parse_strain_gauss_pair() -> None:
    assert benchmark._parse_strain_gauss_pair("5:9") == (5, 9)


@pytest.mark.parametrize("value", ["bad", "-1:5", "1:4"])
def test_parse_strain_gauss_pair_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark._parse_strain_gauss_pair(value)


def test_cpu_requires_single_environment(tmp_path) -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            [
                "--device",
                "cpu",
                "--batch-sizes",
                "1",
                "2",
                "--output",
                str(tmp_path / "result.json"),
            ]
        )


def test_build_system_uses_coupled_shape() -> None:
    system = benchmark._build_system(2, 3, 7, "revolute")

    assert system.num_segments == 2
    assert system.max_num_gauss_points == 7
    assert system.num_dofs == 2 * (1 + 6 * (3 + 1))


def test_build_system_supports_heterogeneous_topologies() -> None:
    alternating = benchmark._build_system(4, 5, 9, "revolute", "alternating")
    grouped = benchmark._build_system(4, 5, 9, "revolute", "grouped")

    expected_dofs = ((1, 24), (1, 36), (1, 24), (1, 36))
    assert alternating.dofs_per_segment_static == expected_dofs
    assert grouped.dofs_per_segment_static == (
        (1, 24),
        (1, 24),
        (1, 36),
        (1, 36),
    )


def test_dynamics_variants_match_values_and_directional_derivatives() -> None:
    system = benchmark._build_system(2, 1, 5, "revolute")
    q, qd = benchmark._batched_inputs(system, 1)
    q, qd = q[0], qd[0]

    def evaluate(variant: str, configuration: jax.Array):
        (
            reuse_velocity,
            use_active_prefix,
            use_compact_basis,
            bucket_count,
            bucket_policy,
            local_shape_policy,
            recurrence_bucket_count,
            recurrence_bucket_policy,
        ) = benchmark._variant_options(system, variant)
        return system._dynamics_terms_impl(
            configuration,
            qd,
            reuse_recurrence_velocity=reuse_velocity,
            use_active_prefix=use_active_prefix,
            use_compact_basis=use_compact_basis,
            reduction_bucket_count=bucket_count,
            reduction_bucket_policy=bucket_policy,
            local_shape_policy=local_shape_policy,
            recurrence_bucket_count=recurrence_bucket_count,
            recurrence_bucket_policy=recurrence_bucket_policy,
        )

    tangent = jnp.linspace(-0.02, 0.02, system.num_dofs)
    expected, expected_jvp = jax.jvp(
        lambda configuration: evaluate("baseline", configuration),
        (q,),
        (tangent,),
    )
    for variant in (
        "velocity",
        "fixed_compact",
        "bucket_uniform_2",
        "bucket_uniform_4",
        "bucket_uniform_8",
        "bucket_equal_4",
        "bucket_optimal_4",
        "local_dofs_4",
        "local_full_4",
        "recurrence_uniform_4",
        "recurrence_uniform_8",
        "active_prefix",
        "compact_basis",
        "optimized",
    ):

        def evaluate_variant(configuration: jax.Array, current_variant: str = variant):
            return evaluate(current_variant, configuration)

        actual, actual_jvp = jax.jvp(
            evaluate_variant,
            (q,),
            (tangent,),
        )
        for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
            assert_allclose(actual_leaf, expected_leaf, rtol=1e-10, atol=1e-11)
        for actual_leaf, expected_leaf in zip(actual_jvp, expected_jvp, strict=True):
            assert_allclose(actual_leaf, expected_leaf, rtol=1e-9, atol=1e-10)


def test_compact_forward_dynamics_benchmark_matches_public_api() -> None:
    system = benchmark._build_system(2, 1, 5, "revolute")
    q, qd = benchmark._batched_inputs(system, 1)
    actual = benchmark._forward_dynamics_callable(system, 1, "compact_basis")(q, qd)
    y = jnp.concatenate([q[0], qd[0]])
    u = jnp.zeros((system.num_actuators,), dtype=q.dtype)
    tau = jnp.zeros((system.num_dofs,), dtype=q.dtype)
    expected = system.forward_dynamics(jnp.array(0.0), y, (u, tau))

    assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


def test_local_shape_variants_match_heterogeneous_baseline() -> None:
    system = benchmark._build_system(2, 3, 7, "revolute", "alternating")
    q, qd = benchmark._batched_inputs(system, 1)
    expected = benchmark._dynamics_callable(system, 1, "baseline")(q, qd)

    for variant in ("local_dofs_4", "local_full_4"):
        actual = benchmark._dynamics_callable(system, 1, variant)(q, qd)
        for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
            assert_allclose(actual_leaf, expected_leaf, rtol=1e-10, atol=1e-11)


def test_single_segment_public_path_matches_fixed_shape_baseline() -> None:
    system = benchmark._build_system(1, 3, 7, "revolute")
    q, qd = benchmark._batched_inputs(system, 1)
    expected = system._dynamics_terms_impl(
        q[0],
        qd[0],
        reuse_recurrence_velocity=False,
        use_active_prefix=False,
        use_compact_basis=False,
    )
    actual = system.dynamics_terms(q[0], qd[0])

    for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
        assert_allclose(actual_leaf, expected_leaf, rtol=0.0, atol=0.0)
