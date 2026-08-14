"""Tests for the reverse-mode-safe numerical primitives.

Each helper is checked twice: it must agree with the naive expression away from
the singular point, and it must stay finite under both autodiff modes at it.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from numpy.testing import assert_allclose

from soromox.autodiff import (
    set_strict_singularities,
    strict_singularities_enabled,
    strict_singularities_mode,
)
from soromox.utils import safe_divide, safe_norm, safe_normalize, safe_sqrt

jax.config.update("jax_enable_x64", True)

ATOL = 1e-12


def assert_forward_and_reverse_mode_finite(fn, arg: Array) -> None:
    """Assert that a JAX function has finite forward and reverse Jacobians."""
    assert jnp.isfinite(jax.jacfwd(fn)(arg)).all()
    assert jnp.isfinite(jax.jacrev(fn)(arg)).all()


class TestSafeSqrt:
    def test_matches_sqrt_away_from_zero(self):
        x = jnp.array([0.25, 1.0, 4.0, 1e6])
        assert_allclose(safe_sqrt(x), jnp.sqrt(x), atol=ATOL)
        assert_allclose(
            jax.grad(lambda v: jnp.sum(safe_sqrt(v)))(x),
            jax.grad(lambda v: jnp.sum(jnp.sqrt(v)))(x),
            atol=ATOL,
        )

    def test_clamps_negative_input_to_zero(self):
        assert float(safe_sqrt(jnp.array(-3.0))) == 0.0

    def test_gradient_is_finite_at_zero(self):
        assert float(jax.grad(safe_sqrt)(jnp.array(0.0))) == 0.0
        assert not jnp.isfinite(jax.grad(jnp.sqrt)(jnp.array(0.0)))

    def test_forward_and_reverse_mode_finite_at_zero(self):
        assert_forward_and_reverse_mode_finite(
            lambda v: safe_sqrt(v).reshape(-1), jnp.zeros(3)
        )


class TestSafeNorm:
    def test_matches_linalg_norm_away_from_zero(self):
        x = jnp.array([3.0, 4.0, 0.0])
        assert_allclose(safe_norm(x), jnp.linalg.norm(x), atol=ATOL)
        assert_allclose(jax.grad(safe_norm)(x), jax.grad(jnp.linalg.norm)(x), atol=ATOL)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_matches_linalg_norm_along_axis(self, axis: int):
        x = jnp.array([[3.0, 4.0], [5.0, 12.0], [1.0, 1.0]])
        assert_allclose(safe_norm(x, axis), jnp.linalg.norm(x, axis=axis), atol=ATOL)

    def test_keepdims_retains_reduced_axis(self):
        x = jnp.array([[3.0, 4.0], [5.0, 12.0]])
        assert safe_norm(x, 1, True).shape == (2, 1)

    def test_gradient_is_finite_at_the_zero_vector(self):
        grad = jax.grad(safe_norm)(jnp.zeros(3))
        assert jnp.isfinite(grad).all()
        assert_allclose(grad, jnp.zeros(3), atol=ATOL)
        # The unguarded norm is the thing being replaced.
        assert jnp.isnan(jax.grad(jnp.linalg.norm)(jnp.zeros(3))).any()

    def test_gradient_is_finite_for_a_zero_row(self):
        x = jnp.array([[3.0, 4.0], [0.0, 0.0]])
        grad = jax.grad(lambda v: jnp.sum(safe_norm(v, 1)))(x)
        assert jnp.isfinite(grad).all()

    def test_hessian_is_finite_at_the_zero_vector(self):
        assert jnp.isfinite(jax.hessian(safe_norm)(jnp.zeros(3))).all()

    def test_finite_under_grad_of_vmap_at_the_zero_vector(self):
        grad = jax.grad(lambda v: jnp.sum(jax.vmap(safe_norm)(v)))(jnp.zeros((4, 3)))
        assert jnp.isfinite(grad).all()

    def test_finite_under_jit(self):
        assert jnp.isfinite(jax.jit(jax.grad(safe_norm))(jnp.zeros(3))).all()


class TestSquaredNormIdentity:
    """``norm(d, axis=k)**2`` equals ``sum(d**2, axis=k)`` but differentiates better.

    This is the substitution applied to the Section Vd losses and to the
    ISupport parallel-axis term, so both halves of the claim are pinned here.
    """

    def test_values_and_gradients_agree_away_from_zero(self):
        d = jnp.array([[1.0, 2.0], [3.0, 4.0], [0.5, -1.5]])
        norm_form = lambda v: jnp.sum(jnp.linalg.norm(v, axis=1) ** 2)  # noqa: E731
        ssq_form = lambda v: jnp.sum(jnp.sum(v**2, axis=1))  # noqa: E731

        assert_allclose(norm_form(d), ssq_form(d), rtol=1e-15)
        assert_allclose(jax.grad(norm_form)(d), jax.grad(ssq_form)(d), atol=ATOL)

    def test_sum_of_squares_survives_an_exactly_zero_row(self):
        d = jnp.array([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]])
        norm_form = lambda v: jnp.sum(jnp.linalg.norm(v, axis=1) ** 2)  # noqa: E731
        ssq_form = lambda v: jnp.sum(jnp.sum(v**2, axis=1))  # noqa: E731

        assert jnp.isnan(jax.grad(norm_form)(d)).any()
        assert jnp.isfinite(jax.grad(ssq_form)(d)).all()


class TestSafeDivide:
    def test_matches_division_away_from_zero(self):
        num = jnp.array([6.0, -3.0])
        den = jnp.array([3.0, 1.5])
        assert_allclose(safe_divide(num, den, 1e-12), num / den, atol=ATOL)

    def test_returns_fallback_at_a_vanishing_denominator(self):
        result = safe_divide(jnp.array(1.0), jnp.array(0.0), 1e-12, fallback=7.0)
        assert float(result) == 7.0

    def test_gradient_is_finite_at_a_vanishing_denominator(self):
        grad = jax.grad(lambda d: safe_divide(jnp.array(1.0), d, 1e-12))(jnp.array(0.0))
        assert jnp.isfinite(grad).all()

    def test_guarding_the_quotient_instead_would_produce_nan(self):
        """Document why the denominator, not the output, must be sanitized."""
        broken = lambda d: jnp.where(d > 0, 1.0 / d, 0.0)  # noqa: E731
        assert jnp.isnan(jax.grad(broken)(jnp.array(0.0))).any()


class TestSafeNormalize:
    def test_matches_naive_normalization_away_from_zero(self):
        x = jnp.array([3.0, 4.0, 0.0])
        assert_allclose(safe_normalize(x), x / jnp.linalg.norm(x), atol=ATOL)

    def test_normalizes_along_an_axis(self):
        x = jnp.array([[3.0, 4.0], [5.0, 12.0]])
        expected = x / jnp.linalg.norm(x, axis=1, keepdims=True)
        assert_allclose(safe_normalize(x, axis=1), expected, atol=ATOL)

    def test_zero_vector_maps_to_the_fallback(self):
        assert_allclose(safe_normalize(jnp.zeros(3)), jnp.zeros(3), atol=ATOL)

    def test_gradient_is_finite_at_the_zero_vector(self):
        grad = jax.grad(lambda v: jnp.sum(safe_normalize(v)))(jnp.zeros(3))
        assert jnp.isfinite(grad).all()

    def test_gradient_is_finite_for_a_zero_row(self):
        x = jnp.array([[3.0, 4.0], [0.0, 0.0]])
        grad = jax.grad(lambda v: jnp.sum(safe_normalize(v, axis=1)))(x)
        assert jnp.isfinite(grad).all()


class TestStrictSingularitiesMode:
    """Opt-in mode that lets singular inputs propagate instead of falling back.

    The fallbacks keep production runs finite but hide *where* a model reaches a
    singular configuration; strict mode trades that back for a diagnostic.
    """

    def test_disabled_by_default(self):
        assert not strict_singularities_enabled()

    @pytest.mark.parametrize(
        "fn, arg",
        [
            (safe_sqrt, jnp.array(0.0)),
            (safe_norm, jnp.zeros(3)),
            (lambda v: jnp.sum(safe_normalize(v)), jnp.zeros(3)),
            (lambda d: safe_divide(jnp.array(1.0), d, 1e-12), jnp.array(0.0)),
        ],
    )
    def test_singular_gradient_propagates_when_strict(self, fn, arg):
        assert jnp.isfinite(jax.grad(fn)(arg)).all()

        with strict_singularities_mode():
            assert not jnp.isfinite(jax.grad(fn)(arg)).all()

    @pytest.mark.parametrize(
        "fn, arg",
        [
            (safe_sqrt, jnp.array(4.0)),
            (safe_norm, jnp.array([3.0, 4.0, 0.0])),
            (lambda v: jnp.sum(safe_normalize(v)), jnp.array([3.0, 4.0, 0.0])),
            (lambda d: safe_divide(jnp.array(6.0), d, 1e-12), jnp.array(3.0)),
        ],
    )
    def test_regular_points_are_unaffected(self, fn, arg):
        value, grad = fn(arg), jax.grad(fn)(arg)

        with strict_singularities_mode():
            assert_allclose(fn(arg), value, atol=ATOL)
            assert_allclose(jax.grad(fn)(arg), grad, atol=ATOL)

    def test_flag_is_restored_after_the_context(self):
        with strict_singularities_mode():
            assert strict_singularities_enabled()
        assert not strict_singularities_enabled()

    def test_flag_is_restored_after_an_exception(self):
        with pytest.raises(RuntimeError), strict_singularities_mode():
            raise RuntimeError("boom")
        assert not strict_singularities_enabled()

    def test_setter_round_trips(self):
        try:
            set_strict_singularities(True)
            assert strict_singularities_enabled()
        finally:
            set_strict_singularities(False)
        assert not strict_singularities_enabled()
