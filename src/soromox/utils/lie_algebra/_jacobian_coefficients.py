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
    """Return the dtype-aware cutoff for inverse left-Jacobian coefficients.

    For an input dtype with machine epsilon ``u``, the cutoff is ``u**(1/8)``.
    It balances the ``O(u / theta**2)`` roundoff amplification in the
    closed-form Hessian of ``(theta / 2) cot(theta / 2)`` against the
    ``O(theta**6)`` Hessian truncation error of the retained inverse series.

    Args:
        dtype: Real floating-point dtype for which to construct the cutoff.

    Returns:
        Scalar array of the requested dtype containing ``u**(1/8)``.

    Notes:
        This is a numerical switching threshold rather than a declaration that
        the mathematical coefficient is singular. The singularity at zero is
        removable, and both branches approximate the same analytic function.
    """
    return jnp.asarray(jnp.finfo(dtype).eps ** (1.0 / 8.0), dtype=dtype)


def forward_left_jacobian_series_threshold(dtype: jnp.dtype) -> Array:
    """Return the dtype-aware cutoff for forward left-Jacobian coefficients.

    For an input dtype with machine epsilon ``u``, the cutoff is ``u**(1/12)``.
    The subtraction-based forward closed forms have Hessian roundoff of order
    ``O(u / theta**4)`` near zero. The series retained in this module have
    Hessian truncation error ``O(theta**8)``; balancing the two errors gives
    ``theta**12 ~= u``.

    Args:
        dtype: Real floating-point dtype for which to construct the cutoff.

    Returns:
        Scalar array of the requested dtype containing ``u**(1/12)``.

    Notes:
        Callers may request a larger cutoff through their ``eps`` argument,
        but never a smaller dtype-aware cutoff.
    """
    return jnp.asarray(jnp.finfo(dtype).eps ** (1.0 / 12.0), dtype=dtype)


def _forward_series_cutoff(theta: Array, eps: float | Array) -> tuple[Array, Array]:
    """Prepare stable branch arguments for a forward scalar coefficient.

    The effective cutoff is
    ``max(abs(eps), forward_left_jacobian_series_threshold(theta.dtype))``.
    Values selected for the Taylor branch are replaced by one before the
    closed-form quotient is evaluated. This sanitization matters under
    ``vmap`` and other transformations that lower scalar conditionals to
    select operations and may evaluate both operands.

    Args:
        theta: Scalar rotation angle.
        eps: User-requested minimum small-angle threshold. Its sign is ignored.

    Returns:
        Tuple ``(use_series, theta_safe)``. ``use_series`` is a scalar Boolean
        array, and ``theta_safe`` equals ``theta`` outside the series region and
        one inside it.
    """
    eps_arr = jnp.abs(jnp.asarray(eps, dtype=theta.dtype))
    threshold = jnp.maximum(
        eps_arr, forward_left_jacobian_series_threshold(theta.dtype)
    )
    use_series = jnp.abs(theta) <= threshold
    theta_safe = jnp.where(use_series, jnp.ones_like(theta), theta)
    return use_series, theta_safe


def sine_over_angle(theta: Array, eps: float | Array = 0.0) -> Array:
    r"""Evaluate the analytic sinc coefficient with stable derivatives.

    The coefficient is

    .. math::

        \operatorname{sinc}(\theta) = \frac{\sin\theta}{\theta}.

    Near zero it is evaluated with the even Taylor polynomial

    .. math::

        1 - \frac{\theta^2}{3!} + \frac{\theta^4}{5!}
        - \frac{\theta^6}{7!} + \frac{\theta^8}{9!}.

    Args:
        theta: Scalar rotation angle. Positive and negative values are
            supported and produce the same coefficient.
        eps: User-requested minimum series cutoff. The effective cutoff is the
            maximum of ``abs(eps)`` and the dtype-aware forward cutoff.

    Returns:
        Scalar array with the continuous value one at ``theta == 0`` in the
        default mode.

    Notes:
        In :func:`soromox.autodiff.strict_singularities_mode`, the raw quotient
        is returned and therefore produces a non-finite value at zero. Outside
        strict mode the closed-form input is sanitized before branch selection,
        keeping values, Jacobians, and Hessians finite under JAX transforms.
    """
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
    r"""Evaluate the analytic cosc coefficient with stable derivatives.

    The coefficient is

    .. math::

        \operatorname{cosc}(\theta)
        = \frac{1 - \cos\theta}{\theta^2},

    with Taylor polynomial

    .. math::

        \frac{1}{2!} - \frac{\theta^2}{4!}
        + \frac{\theta^4}{6!} - \frac{\theta^6}{8!}
        + \frac{\theta^8}{10!}

    in the small-angle region. This coefficient multiplies the squared skew
    matrix in Rodrigues' formula and the first skew term of the ``SE(3)`` left
    Jacobian.

    Args:
        theta: Scalar rotation angle.
        eps: User-requested minimum series cutoff. The dtype-aware forward
            cutoff remains a lower bound.

    Returns:
        Scalar array with the continuous value ``1 / 2`` at zero in default
        mode.

    Notes:
        Strict-singularity mode evaluates the unsanitized quotient, exposing
        its removable ``0 / 0`` form at zero for diagnostics.
    """
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
    r"""Evaluate the cubic forward-Jacobian coefficient stably.

    The coefficient is

    .. math::

        \operatorname{tanc}(\theta)
        = \frac{\theta - \sin\theta}{\theta^3},

    with Taylor polynomial

    .. math::

        \frac{1}{3!} - \frac{\theta^2}{5!}
        + \frac{\theta^4}{7!} - \frac{\theta^6}{9!}
        + \frac{\theta^8}{11!}.

    It is used as the quadratic-skew coefficient of the ``SE(3)`` forward left
    Jacobian.

    Args:
        theta: Scalar rotation angle.
        eps: User-requested minimum series cutoff. The dtype-aware forward
            cutoff remains a lower bound.

    Returns:
        Scalar array with the continuous value ``1 / 6`` at zero in default
        mode.

    Notes:
        Strict-singularity mode returns the raw quotient and intentionally
        exposes the removable singularity at zero.
    """
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
    r"""Evaluate the inverse-Jacobian half-angle coefficient stably.

    The analytic coefficient

    .. math::

        h(\theta) = \frac{\theta}{2}\cot\!\left(\frac{\theta}{2}\right)

    is evaluated near zero as

    .. math::

        1 - \frac{\theta^2}{12} - \frac{\theta^4}{720}
        - \frac{\theta^6}{30240}.

    Args:
        theta: Scalar rotation angle.

    Returns:
        Scalar array with the continuous value one at zero in default mode.

    Notes:
        The inverse cutoff ``u**(1/8)`` differs intentionally from the forward
        cutoff because the cancellation and retained series orders differ.
        Strict-singularity mode evaluates the raw half-angle quotient.
    """
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
    r"""Evaluate the quadratic coefficient of the inverse ``SE(3)`` Jacobian.

    For ``Omega = skew(omega)`` and ``theta = norm(omega)``, the inverse left
    Jacobian is

    .. math::

        J_l^{-1}(\omega) = I - \frac{1}{2}\Omega + q(\theta)\Omega^2,

    where

    .. math::

        q(\theta) =
        \frac{1 - (\theta/2)\cot(\theta/2)}{\theta^2}.

    The small-angle branch uses

    .. math::

        \frac{1}{12} + \frac{\theta^2}{720}
        + \frac{\theta^4}{30240} + \frac{\theta^6}{1209600}.

    Args:
        theta: Scalar rotation magnitude.

    Returns:
        Scalar array with the continuous value ``1 / 12`` at zero in default
        mode.

    Notes:
        The closed form contains both a cotangent quotient and an outer
        subtraction/division, so its input is sanitized before selection.
        Strict-singularity mode restores the complete raw expression.
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
