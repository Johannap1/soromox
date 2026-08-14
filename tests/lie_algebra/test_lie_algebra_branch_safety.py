"""Reverse-mode safety of the Lie-algebra small-angle branches.

The small-angle guards in ``so3``, ``se3`` and ``constant_strain`` use
``lax.cond`` deliberately, and this file exists to keep it that way.

Under reverse-mode autodiff ``lax.cond`` differentiates only the taken branch,
including under ``vmap``: JAX lowers a batched predicate to ``select_n`` but
still routes a symbolic zero cotangent into the branch that was not taken, so
the singular closed-form expression never reaches the gradient. ``jnp.where``
and ``lax.select`` do *not* have that property -- their transpose multiplies the
dead branch by a materialized zero, and ``0 * NaN`` is ``NaN``.

Rewriting any of these guards as ``jnp.where`` would therefore introduce exactly
the NaN it looks like it prevents. These tests fail loudly if that happens.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from soromox.utils.lie_algebra import constant_strain, se3, so3

jax.config.update("jax_enable_x64", True)

GLOBAL_EPS = 1e1 * float(jnp.finfo(jnp.float64).eps)
TANGENT_EPS = float(jnp.sqrt(jnp.asarray(GLOBAL_EPS)))

# Straight backbone: zero rotational strain, unit elongation.
XI_STRAIGHT = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
ARC_LENGTH = jnp.array(0.1)


def _batched_grad(fn, x: Array, batch: int = 3) -> Array:
    """Differentiate ``sum(vmap(fn)(x))``, i.e. grad on the outside of vmap.

    This is the transform order the PCS/GVS recurrences actually produce, and it
    is the one that lowers ``lax.cond`` into ``select_n``.
    """
    stacked = jnp.broadcast_to(x, (batch,) + x.shape)
    return jax.grad(lambda v: jnp.sum(jax.vmap(fn)(v)))(stacked)


LIE_CASES = {
    "se3.exp": (lambda xi: se3.exp(xi, GLOBAL_EPS), XI_STRAIGHT),
    "so3.exp": (lambda xi: so3.exp(xi[:3], GLOBAL_EPS), XI_STRAIGHT),
    "constant_strain.tangent_se3": (
        lambda xi: constant_strain.tangent_se3(xi, ARC_LENGTH, TANGENT_EPS),
        XI_STRAIGHT,
    ),
    "constant_strain.adjoint_se3": (
        lambda xi: constant_strain.adjoint_se3(xi, ARC_LENGTH, GLOBAL_EPS),
        XI_STRAIGHT,
    ),
    "constant_strain.adjoint_inverse_se3": (
        lambda xi: constant_strain.adjoint_inverse_se3(xi, ARC_LENGTH, GLOBAL_EPS),
        XI_STRAIGHT,
    ),
    "constant_strain.tangent_derivative_se3": (
        lambda xi: constant_strain.tangent_derivative_se3(
            xi, jnp.zeros(6), ARC_LENGTH, TANGENT_EPS
        ),
        XI_STRAIGHT,
    ),
    "se3.log": (lambda g: se3.log(g, GLOBAL_EPS), jnp.eye(4)),
    "so3.log": (lambda R: so3.log(R, GLOBAL_EPS), jnp.eye(3)),
}


@pytest.mark.parametrize("name", sorted(LIE_CASES))
def test_grad_of_vmap_is_finite_at_zero_rotation(name: str):
    """The batched lowering rewrites ``lax.cond`` into ``select_n``.

    Only the batched order is checked: unbatched, ``lax.cond`` stays a ``cond``
    and is the easier case, so an unbatched gradient can only be non-finite if
    the batched one is too. The unbatched path is also already covered by the
    ``*_at_zero_configuration`` tests in ``tests/systems``.
    """
    fn, arg = LIE_CASES[name]

    grad = _batched_grad(lambda x: jnp.sum(fn(x)), arg)

    assert jnp.isfinite(grad).all(), f"{name} produced a non-finite batched gradient"


def test_where_would_regress_what_cond_gets_right():
    """Pin the property the Lie-algebra guards rely on.

    If a future JAX release makes ``lax.cond`` behave like ``jnp.where`` here,
    this test fails and the guards must be revisited rather than silently
    starting to emit NaN.
    """

    def singular(x: Array) -> Array:
        return (1.0 - jnp.cos(x)) / x**2  # 0 / 0 exactly at x == 0

    def with_cond(x: Array) -> Array:
        return jax.lax.cond(
            x <= 0.0, lambda _: jnp.zeros_like(x), lambda _: singular(x), operand=None
        )

    def with_where(x: Array) -> Array:
        return jnp.where(x <= 0.0, jnp.zeros_like(x), singular(x))

    zeros = jnp.zeros(2)
    cond_grad = jax.grad(lambda v: jnp.sum(jax.vmap(with_cond)(v)))(zeros)
    where_grad = jax.grad(lambda v: jnp.sum(jax.vmap(with_where)(v)))(zeros)

    assert jnp.isfinite(cond_grad).all(), "lax.cond must not leak the dead branch"
    assert jnp.isnan(where_grad).any(), (
        "jnp.where is expected to leak the dead branch; if this now passes, the "
        "safety argument in utils/_numerics.py needs revisiting"
    )
