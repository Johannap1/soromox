"""Stable scalar coefficients for inverse Lie-group left Jacobians.

The inverse ``SE(2)`` and ``SE(3)`` left Jacobians contain removable
singularities at zero rotation.  Their closed forms are accurate away from the
origin, while Taylor series are needed near it for accurate reverse- and
higher-order derivatives.

The shared series cutoff is derived for second-order autodiff, with ``u``
denoting machine epsilon for the input dtype.  The closed-form Hessian of
``(theta / 2) cot(theta / 2)`` amplifies roundoff like ``O(u / theta**2)``.
The series retained here has Hessian truncation error ``O(theta**6)``.  Balancing
those errors gives ``theta**8 ~= u``, hence the common cutoff ``u**(1/8)``.
This threshold is also conservative enough for the additional subtraction and
division in the ``SE(3)`` quadratic coefficient.

Both helpers use the same policy: select a sufficiently high-order series near
zero, sanitize the closed-form input so batched selection cannot evaluate a
singular dead branch, and restore the unsanitized quotient under
``strict_singularities_mode``.
"""

__all__ = [
    "half_angle_cotangent",
    "inverse_left_jacobian_quadratic_coefficient",
    "inverse_left_jacobian_series_threshold",
]

import jax.numpy as jnp
from jax import Array

from soromox.autodiff import strict_singularities_enabled


def inverse_left_jacobian_series_threshold(dtype: jnp.dtype) -> Array:
    """Return the shared angle cutoff for inverse-left-Jacobian series."""
    return jnp.asarray(jnp.finfo(dtype).eps ** (1.0 / 8.0), dtype=dtype)


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
