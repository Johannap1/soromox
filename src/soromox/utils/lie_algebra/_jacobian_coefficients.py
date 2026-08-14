"""Stable scalar coefficients used by Lie-group Jacobians and maps.

Forward and inverse Lie-group Jacobians contain removable singularities at zero
rotation. Their closed forms are accurate away from the origin, while Taylor
series are needed near it for accurate reverse- and higher-order derivatives.

The cutoffs are derived for second-order autodiff, with ``u`` denoting machine
epsilon for the input dtype. The closed-form Hessian of
``(theta / 2) cot(theta / 2)`` amplifies roundoff like ``O(u / theta**2)``.
The inverse series has Hessian truncation error ``O(theta**6)``. Balancing
those errors gives ``theta**8 ~= u``, hence the common cutoff ``u**(1/8)``.
This threshold is also conservative enough for the additional subtraction and
division in the ``SE(3)`` quadratic coefficient.

The forward coefficients retain terms through ``theta**8``. The Hessians of
their subtraction-based closed forms amplify roundoff like ``O(u / theta**4)``,
while the series Hessian truncation is ``O(theta**8)``. Balancing the two gives
``theta**12 ~= u`` and the forward cutoff ``u**(1/12)``.

All helpers use the same policy: select a sufficiently high-order series near
zero, sanitize the closed-form input so batched selection cannot evaluate a
singular dead branch, and restore the unsanitized quotients under
``strict_singularities_mode``.
"""

import jax.numpy as jnp
from jax import Array

from soromox.autodiff import strict_singularities_enabled


def inverse_left_jacobian_series_threshold(dtype: jnp.dtype) -> Array:
    """Return the shared angle cutoff for inverse-left-Jacobian series."""
    return jnp.asarray(jnp.finfo(dtype).eps ** (1.0 / 8.0), dtype=dtype)


def forward_left_jacobian_series_threshold(dtype: jnp.dtype) -> Array:
    """Return the angle cutoff for forward-left-Jacobian series."""
    return jnp.asarray(jnp.finfo(dtype).eps ** (1.0 / 12.0), dtype=dtype)


def _forward_series_cutoff(theta: Array, eps: float | Array) -> tuple[Array, Array]:
    """Return the forward-series predicate and a sanitized closed-form input."""
    eps_arr = jnp.abs(jnp.asarray(eps, dtype=theta.dtype))
    threshold = jnp.maximum(
        eps_arr, forward_left_jacobian_series_threshold(theta.dtype)
    )
    use_series = jnp.abs(theta) <= threshold
    theta_safe = jnp.where(use_series, jnp.ones_like(theta), theta)
    return use_series, theta_safe


def sine_over_angle(theta: Array, eps: float | Array = 0.0) -> Array:
    """Evaluate ``sin(theta) / theta`` with stable second derivatives."""
    theta = jnp.asarray(theta)
    if strict_singularities_enabled():
        return jnp.sin(theta) / theta

    theta_sq = theta**2
    series = 1.0 + theta_sq * (
        -1.0 / 6.0
        + theta_sq * (1.0 / 120.0 + theta_sq * (-1.0 / 5040.0 + theta_sq / 362880.0))
    )
    use_series, theta_safe = _forward_series_cutoff(theta, eps)
    closed = jnp.sin(theta_safe) / theta_safe
    return jnp.where(use_series, series, closed)


def one_minus_cosine_over_angle_squared(
    theta: Array, eps: float | Array = 0.0
) -> Array:
    """Evaluate ``(1 - cos(theta)) / theta**2`` with stable derivatives."""
    theta = jnp.asarray(theta)
    if strict_singularities_enabled():
        return (1.0 - jnp.cos(theta)) / theta**2

    theta_sq = theta**2
    series = 0.5 + theta_sq * (
        -1.0 / 24.0
        + theta_sq * (1.0 / 720.0 + theta_sq * (-1.0 / 40320.0 + theta_sq / 3628800.0))
    )
    use_series, theta_safe = _forward_series_cutoff(theta, eps)
    closed = (1.0 - jnp.cos(theta_safe)) / theta_safe**2
    return jnp.where(use_series, series, closed)


def angle_minus_sine_over_angle_cubed(theta: Array, eps: float | Array = 0.0) -> Array:
    """Evaluate ``(theta - sin(theta)) / theta**3`` stably."""
    theta = jnp.asarray(theta)
    if strict_singularities_enabled():
        return (theta - jnp.sin(theta)) / theta**3

    theta_sq = theta**2
    series = 1.0 / 6.0 + theta_sq * (
        -1.0 / 120.0
        + theta_sq
        * (1.0 / 5040.0 + theta_sq * (-1.0 / 362880.0 + theta_sq / 39916800.0))
    )
    use_series, theta_safe = _forward_series_cutoff(theta, eps)
    closed = (theta_safe - jnp.sin(theta_safe)) / theta_safe**3
    return jnp.where(use_series, series, closed)


def half_angle_cotangent(theta: Array) -> Array:
    """Evaluate ``(theta / 2) cot(theta / 2)`` with stable derivatives."""
    theta = jnp.asarray(theta)
    half_theta = 0.5 * theta

    if strict_singularities_enabled():
        return half_theta / jnp.tan(half_theta)

    theta_sq = theta**2
    series = 1.0 - theta_sq / 12.0 - theta_sq**2 / 720.0 - theta_sq**3 / 30240.0
    threshold = inverse_left_jacobian_series_threshold(theta.dtype)
    use_series = jnp.abs(theta) <= threshold

    theta_safe = jnp.where(use_series, jnp.ones_like(theta), theta)
    half_theta_safe = 0.5 * theta_safe
    closed = half_theta_safe / jnp.tan(half_theta_safe)
    return jnp.where(use_series, series, closed)


def inverse_left_jacobian_quadratic_coefficient(theta: Array) -> Array:
    """Evaluate the ``Omega**2`` coefficient of the inverse ``SE(3)`` Jacobian.

    The returned value is
    ``(1 - (theta / 2) cot(theta / 2)) / theta**2`` with its analytic series
    used at small angles.
    """
    theta = jnp.asarray(theta)
    theta_sq = theta**2
    half_theta = 0.5 * theta

    if strict_singularities_enabled():
        return (1.0 - half_theta / jnp.tan(half_theta)) / theta_sq

    series = (
        1.0 / 12.0 + theta_sq / 720.0 + theta_sq**2 / 30240.0 + theta_sq**3 / 1209600.0
    )
    threshold = inverse_left_jacobian_series_threshold(theta.dtype)
    use_series = jnp.abs(theta) <= threshold

    theta_safe = jnp.where(use_series, jnp.ones_like(theta), theta)
    half_theta_safe = 0.5 * theta_safe
    closed = (1.0 - half_theta_safe / jnp.tan(half_theta_safe)) / theta_safe**2
    return jnp.where(use_series, series, closed)
