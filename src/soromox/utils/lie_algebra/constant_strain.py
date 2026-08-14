__all__ = [
    "adjoint_se2",
    "adjoint_inverse_se2",
    "tangent_se2",
    "tangent_derivative_se2",
    "adjoint_se3",
    "adjoint_inverse_se3",
    "tangent_se3",
    "tangent_derivative_se3",
]

import jax.numpy as jnp
from jax import Array

from soromox.autodiff import strict_singularities_enabled

from . import se2, se3


def _constant_strain_series_threshold(dtype: jnp.dtype) -> Array:
    """Return the dimensionless cutoff for constant-strain scalar series.

    The worst closed-form Hessian roundoff grows as ``O(u / z**6)`` for
    ``z = s * theta``. The retained scalar series have Hessian truncation
    ``O(z**8)``. Balancing the errors yields ``z**14 ~= u``.
    """
    return jnp.asarray(jnp.finfo(dtype).eps ** (1.0 / 14.0), dtype=dtype)


def _series_arguments(
    angle_sq: Array, s: Array, eps: float | Array
) -> tuple[Array, Array, Array]:
    """Return ``x=(s*theta)^2``, the series predicate, and a safe ``sqrt(x)``."""
    x = s**2 * angle_sq
    requested = jnp.abs(s) * jnp.abs(jnp.asarray(eps, dtype=x.dtype))
    threshold = jnp.maximum(requested, _constant_strain_series_threshold(x.dtype))
    use_series = x <= threshold**2
    x_safe = jnp.where(use_series, jnp.ones_like(x), x)
    return x, use_series, jnp.sqrt(x_safe)


def _adjoint_coefficients(
    angle_sq: Array, s: Array, eps: float | Array
) -> tuple[Array, Array, Array, Array]:
    """Return the four stable coefficients of the reduced adjoint polynomial."""
    x, use_series, z_safe = _series_arguments(angle_sq, s, eps)
    series = (
        1.0 + x**2 * (-1.0 / 120.0 + x * (1.0 / 2520.0 - x / 120960.0)),
        0.5 + x**2 * (-1.0 / 720.0 + x * (1.0 / 20160.0 - x / 1209600.0)),
        1.0 / 6.0
        + x * (-1.0 / 60.0 + x * (1.0 / 1680.0 + x * (-1.0 / 90720.0 + x / 7983360.0))),
        1.0 / 24.0
        + x
        * (-1.0 / 360.0 + x * (1.0 / 13440.0 + x * (-1.0 / 907200.0 + x / 95800320.0))),
    )
    if strict_singularities_enabled():
        z_safe = jnp.sqrt(x)
        use_series = jnp.zeros_like(x, dtype=jnp.bool_)
    sin_z = jnp.sin(z_safe)
    cos_z = jnp.cos(z_safe)
    closed = (
        (3.0 * sin_z - z_safe * cos_z) / (2.0 * z_safe),
        (4.0 - 4.0 * cos_z - z_safe * sin_z) / (2.0 * z_safe**2),
        (sin_z - z_safe * cos_z) / (2.0 * z_safe**3),
        (2.0 - 2.0 * cos_z - z_safe * sin_z) / (2.0 * z_safe**4),
    )
    return tuple(
        jnp.where(use_series, a, b) for a, b in zip(series, closed, strict=True)
    )


def _tangent_coefficients_and_x_derivatives(
    angle_sq: Array, s: Array, eps: float | Array
) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
    """Return tangent coefficients and analytic derivatives with respect to ``x``."""
    x, use_series, z_safe = _series_arguments(angle_sq, s, eps)
    series = (
        0.5 + x**2 * (-1.0 / 720.0 + x * (1.0 / 20160.0 - x / 1209600.0)),
        1.0 / 6.0 + x**2 * (-1.0 / 5040.0 + x * (1.0 / 181440.0 - x / 13305600.0)),
        1.0 / 24.0
        + x
        * (-1.0 / 360.0 + x * (1.0 / 13440.0 + x * (-1.0 / 907200.0 + x / 95800320.0))),
        1.0 / 120.0
        + x
        * (
            -1.0 / 2520.0
            + x * (1.0 / 120960.0 + x * (-1.0 / 9979200.0 + x / 1245404160.0))
        ),
    )
    derivative_series = (
        x * (-1.0 / 360.0 + x * (1.0 / 6720.0 - x / 302400.0)),
        x * (-1.0 / 2520.0 + x * (1.0 / 60480.0 - x / 3326400.0)),
        -1.0 / 360.0 + x * (1.0 / 6720.0 + x * (-1.0 / 302400.0 + x / 23950080.0)),
        -1.0 / 2520.0 + x * (1.0 / 60480.0 + x * (-1.0 / 3326400.0 + x / 311351040.0)),
    )
    if strict_singularities_enabled():
        z_safe = jnp.sqrt(x)
        use_series = jnp.zeros_like(x, dtype=jnp.bool_)
    sin_z = jnp.sin(z_safe)
    cos_z = jnp.cos(z_safe)
    common = -8.0 + (8.0 - z_safe**2) * cos_z + 5.0 * z_safe * sin_z
    alternate = -8.0 * z_safe + (15.0 - z_safe**2) * sin_z - 7.0 * z_safe * cos_z
    closed = (
        (4.0 - 4.0 * cos_z - z_safe * sin_z) / (2.0 * z_safe**2),
        (4.0 * z_safe - 5.0 * sin_z + z_safe * cos_z) / (2.0 * z_safe**3),
        (2.0 - 2.0 * cos_z - z_safe * sin_z) / (2.0 * z_safe**4),
        (2.0 * z_safe - 3.0 * sin_z + z_safe * cos_z) / (2.0 * z_safe**5),
    )
    derivative_closed = (
        common / (4.0 * z_safe**4),
        alternate / (4.0 * z_safe**5),
        common / (4.0 * z_safe**6),
        alternate / (4.0 * z_safe**7),
    )
    coefficients = tuple(
        jnp.where(use_series, a, b) for a, b in zip(series, closed, strict=True)
    )
    derivatives = tuple(
        jnp.where(use_series, a, b)
        for a, b in zip(derivative_series, derivative_closed, strict=True)
    )
    return coefficients, derivatives


def _matrix_powers(matrix: Array) -> list[Array]:
    """Return ``[I, matrix, ..., matrix**4]`` with shared multiplications."""
    powers = [jnp.eye(matrix.shape[0], dtype=matrix.dtype)]
    for _ in range(4):
        powers.append(powers[-1] @ matrix)
    return powers


def _matrix_powers_with_derivatives(
    matrix: Array, matrix_dot: Array
) -> tuple[list[Array], list[Array]]:
    """Return fourth-order matrix powers and explicit directional derivatives."""
    current = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    dot_current = jnp.zeros_like(matrix)
    powers = [current]
    dot_powers = [dot_current]
    for _ in range(4):
        dot_next = dot_current @ matrix + current @ matrix_dot
        current = current @ matrix
        powers.append(current)
        dot_powers.append(dot_next)
        dot_current = dot_next
    return powers, dot_powers


def _constant_strain_adjoint(
    ad_xi: Array, angle_sq: Array, s: Array, eps: float | Array
) -> Array:
    """Evaluate the stable fourth-order constant-strain adjoint polynomial."""
    powers = _matrix_powers(s * ad_xi)
    result = powers[0]
    for coefficient, power in zip(
        _adjoint_coefficients(angle_sq, s, eps), powers[1:], strict=True
    ):
        result = result + coefficient * power
    return result


def _constant_strain_tangent(
    ad_xi: Array, angle_sq: Array, s: Array, eps: float | Array
) -> Array:
    """Evaluate the stable fourth-order constant-strain tangent polynomial."""
    powers = _matrix_powers(s * ad_xi)
    coefficients, _ = _tangent_coefficients_and_x_derivatives(angle_sq, s, eps)
    result = powers[0]
    for coefficient, power in zip(coefficients, powers[1:], strict=True):
        result = result + coefficient * power
    return s * result


def _constant_strain_tangent_derivative(
    ad_xi: Array,
    ad_xid: Array,
    angle_sq: Array,
    angle_sq_dot: Array,
    s: Array,
    eps: float | Array,
) -> Array:
    """Evaluate the tangent's explicit analytic directional derivative."""
    powers, dot_powers = _matrix_powers_with_derivatives(s * ad_xi, s * ad_xid)
    coefficients, derivatives = _tangent_coefficients_and_x_derivatives(
        angle_sq, s, eps
    )
    x_dot = s**2 * angle_sq_dot
    result = jnp.zeros_like(ad_xi)
    for coefficient, derivative, power, dot_power in zip(
        coefficients, derivatives, powers[1:], dot_powers[1:], strict=True
    ):
        result = result + derivative * x_dot * power + coefficient * dot_power
    return s * result


def adjoint_se2(xi: Array, s: Array, eps: float | Array) -> Array:
    """Return the adjoint accumulated along a constant planar strain segment.

    This is a rod-kinematics helper, not a generic transform adjoint. For a
    segment with constant planar strain ``xi`` evaluated at arclength ``s``,
    the returned matrix represents the closed-form adjoint associated with the
    segment transform ``exp(s * xi)`` under the angular-first ``se(2)``
    convention.

    Args:
        xi: Constant planar strain with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold. The actual dimensionless
            cutoff also accounts for ``s`` and the input dtype.

    Returns:
        Array with shape ``(3, 3)`` mapping planar twists through the segment
        adjoint at arclength ``s``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    angle_sq = xi[0] ** 2
    return _constant_strain_adjoint(se2.small_adjoint(xi), angle_sq, s, eps)


def adjoint_inverse_se2(xi: Array, s: Array, eps: float | Array) -> Array:
    """Return the inverse adjoint for a constant planar strain segment.

    This computes the inverse of :func:`adjoint_se2` without explicitly
    inverting the full homogeneous segment transform. It is used when
    propagating planar body-frame Jacobians backward through a constant-strain
    segment.

    Args:
        xi: Constant planar strain with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Small positive scalar threshold for the same small-angle branch
            used by :func:`adjoint_se2`.

    Returns:
        Array with shape ``(3, 3)`` representing the inverse segment adjoint.
    """
    Ad = adjoint_se2(xi, s, eps=eps)

    R = Ad[1:, 1:]
    mJt = Ad[1:, 0].reshape(-1, 1)
    R_inv = jnp.transpose(R)

    return jnp.concatenate(
        [
            jnp.concatenate(
                [jnp.ones((1, 1), dtype=Ad.dtype), jnp.zeros((1, 2), dtype=Ad.dtype)],
                axis=1,
            ),
            jnp.concatenate([-R_inv @ mJt, R_inv], axis=1),
        ]
    )


def tangent_se2(xi: Array, s: Array, eps: float | Array) -> Array:
    """Return the tangent operator for a constant planar strain segment.

    The tangent operator integrates the local strain perturbation over
    arclength for a segment whose strain is constant. In PCS/GVS recurrences it
    maps strain-basis contributions into body-frame Jacobian contributions at
    arclength ``s``.

    Args:
        xi: Constant planar strain with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold. The actual dimensionless
            cutoff also accounts for ``s`` and the input dtype.

    Returns:
        Array with shape ``(3, 3)`` containing the planar constant-strain
        tangent operator at arclength ``s``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    angle_sq = xi[0] ** 2
    return _constant_strain_tangent(se2.small_adjoint(xi), angle_sq, s, eps)


def tangent_derivative_se2(
    xi: Array, xid: Array, s: Array, eps: float | Array
) -> Array:
    """Return the time derivative of the planar tangent operator.

    This differentiates :func:`tangent_se2` with respect to time through the
    strain ``xi`` and strain rate ``xid`` while holding arclength ``s`` fixed.
    The function uses the same angular-first planar convention as ``se2``.

    Args:
        xi: Constant planar strain with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.
        xid: Time derivative of ``xi`` with the same shape and coordinate
            order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold. The actual dimensionless
            cutoff also accounts for ``s`` and the input dtype.

    Returns:
        Array with shape ``(3, 3)`` containing ``d/dt tangent_se2(xi, s)``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    xid = jnp.asarray(xid).reshape(-1)
    angle_sq = xi[0] ** 2
    angle_sq_dot = 2.0 * xi[0] * xid[0]
    return _constant_strain_tangent_derivative(
        se2.small_adjoint(xi),
        se2.small_adjoint(xid),
        angle_sq,
        angle_sq_dot,
        s,
        eps,
    )


def adjoint_se3(xi: Array, s: Array, eps: float | Array) -> Array:
    """Return the adjoint accumulated along a constant spatial strain segment.

    This is a rod-kinematics helper for spatial PCS/GVS segments. For constant
    strain ``xi`` evaluated at arclength ``s``, the returned matrix represents
    the closed-form adjoint associated with the segment transform
    ``exp(s * xi)`` under the angular-first ``se(3)`` convention.

    Args:
        xi: Constant spatial strain with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold. The actual dimensionless
            cutoff also accounts for ``s`` and the input dtype.

    Returns:
        Array with shape ``(6, 6)`` mapping spatial twists through the segment
        adjoint at arclength ``s``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    angle_sq = jnp.dot(xi[:3], xi[:3])
    return _constant_strain_adjoint(se3.small_adjoint(xi), angle_sq, s, eps)


def adjoint_inverse_se3(xi: Array, s: Array, eps: float | Array) -> Array:
    """Return the inverse adjoint for a constant spatial strain segment.

    This computes the inverse of :func:`adjoint_se3` without explicitly
    constructing and inverting the homogeneous segment transform. It is used
    when propagating body-frame Jacobians backward through spatial
    constant-strain segments.

    Args:
        xi: Constant spatial strain with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Small positive scalar threshold for the same small-angle branch
            used by :func:`adjoint_se3`.

    Returns:
        Array with shape ``(6, 6)`` representing the inverse segment adjoint.
    """
    Ad = adjoint_se3(xi, s, eps=eps)

    R = Ad[:3, :3]
    t_hat_R = Ad[3:, :3]
    R_inv = jnp.transpose(R)
    t_hat = t_hat_R @ R_inv

    return jnp.block(
        [[R_inv, jnp.zeros((3, 3), dtype=Ad.dtype)], [-R_inv @ t_hat, R_inv]]
    )


def tangent_se3(xi: Array, s: Array, eps: float | Array) -> Array:
    """Return the tangent operator for a constant spatial strain segment.

    The tangent operator integrates local spatial strain perturbations over
    arclength for a segment whose strain is constant. In PCS/GVS recurrences it
    maps strain-basis contributions into body-frame Jacobian contributions at
    arclength ``s``.

    Args:
        xi: Constant spatial strain with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold. The actual dimensionless
            cutoff also accounts for ``s`` and the input dtype.

    Returns:
        Array with shape ``(6, 6)`` containing the spatial constant-strain
        tangent operator at arclength ``s``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    angle_sq = jnp.dot(xi[:3], xi[:3])
    return _constant_strain_tangent(se3.small_adjoint(xi), angle_sq, s, eps)


def tangent_derivative_se3(
    xi: Array, xid: Array, s: Array, eps: float | Array
) -> Array:
    """Return the time derivative of the spatial tangent operator.

    This differentiates :func:`tangent_se3` with respect to time through the
    strain ``xi`` and strain rate ``xid`` while holding arclength ``s`` fixed.
    The coefficient derivative is expressed through the squared rotational
    magnitude, avoiding a division by ``norm(omega)`` at zero rotation.

    Args:
        xi: Constant spatial strain with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        xid: Time derivative of ``xi`` with the same shape and coordinate
            order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold. The actual dimensionless
            cutoff also accounts for ``s`` and the input dtype.

    Returns:
        Array with shape ``(6, 6)`` containing ``d/dt tangent_se3(xi, s)``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    xid = jnp.asarray(xid).reshape(-1)
    angle_sq = jnp.dot(xi[:3], xi[:3])
    angle_sq_dot = 2.0 * jnp.dot(xi[:3], xid[:3])
    return _constant_strain_tangent_derivative(
        se3.small_adjoint(xi),
        se3.small_adjoint(xid),
        angle_sq,
        angle_sq_dot,
        s,
        eps,
    )
