r"""Closed-form Lie operators for constant-strain rod segments.

For a constant body strain ``xi`` and arclength ``s``, let
``A = ad_xi`` and ``B = s A``. This module evaluates three related operators:

.. math::

    \operatorname{Ad}(s, \xi) &= \exp(B), \\
    T(s, \xi) &= \int_0^s \exp(\sigma A)\,d\sigma, \\
    \dot T(s, \xi, \dot\xi)
        &= D_\xi T(s, \xi)[\dot\xi].

The minimal polynomials of ``se(2)`` and ``se(3)`` adjoint matrices reduce all
three expressions exactly to powers through ``B**4``. Their scalar
trigonometric coefficients nevertheless contain removable singularities at
zero rotation. The implementation therefore uses closed forms away from zero
and even Taylor series through ``z**8`` near zero, where ``z = s * theta``.
The coefficients are expressed through ``x = z**2`` so neither the primal nor
the explicit tangent derivative divides by ``theta = norm(omega)``.

The default branch policy is designed to keep values, reverse-mode gradients,
and Hessians finite under scalar and batched JAX transformations. Under
``strict_singularities_mode`` the raw quotients are restored for diagnostics.
No automatic-differentiation operation is used in these production routines;
``tangent_derivative_se2`` and ``tangent_derivative_se3`` are explicit analytic
directional derivatives.
"""

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
    """Return the dtype-aware dimensionless constant-strain cutoff.

    The worst closed-form Hessian roundoff grows as ``O(u / z**6)`` for
    ``z = s * theta``. The retained scalar series have Hessian truncation
    ``O(z**8)``. Balancing the errors yields ``z**14 ~= u``.

    Args:
        dtype: Real floating-point dtype for which to construct the cutoff.

    Returns:
        Scalar array of the requested dtype containing ``u**(1/14)``, where
        ``u`` is the dtype's machine epsilon.

    Notes:
        This dimensionless cutoff applies to the accumulated rotation ``z``,
        not directly to the rotational strain ``theta``. It therefore remains
        meaningful for negative ``s`` and becomes independent of the units
        used to parameterize arclength.
    """
    return jnp.asarray(jnp.finfo(dtype).eps ** (1.0 / 14.0), dtype=dtype)


def _series_arguments(
    angle_sq: Array, s: Array, eps: float | Array
) -> tuple[Array, Array, Array]:
    """Prepare stable scalar arguments for constant-strain coefficients.

    The coefficient functions are even in ``z = s * theta`` and are therefore
    represented using ``x = z**2 = s**2 * angle_sq``. The effective
    dimensionless cutoff is ``max(abs(s) * abs(eps), u**(1/14))``. In the
    series region ``x`` is replaced by one before the regular closed forms take
    a square root or divide by powers of ``z``. This prevents a singular dead
    branch from contaminating derivatives after ``vmap`` lowers selection.

    Args:
        angle_sq: Squared rotational strain. For ``se(2)`` this is
            ``xi[0]**2``; for ``se(3)`` it is ``dot(omega, omega)``.
        s: Scalar arclength, which may be positive, negative, or zero.
        eps: User-requested minimum threshold in rotational-strain units.

    Returns:
        Tuple ``(x, use_series, z_safe)`` containing the squared accumulated
        rotation, the scalar series predicate, and a sanitized nonnegative
        magnitude ``sqrt(x_safe)`` for closed-form evaluation.
    """
    x = s**2 * angle_sq
    requested = jnp.abs(s) * jnp.abs(jnp.asarray(eps, dtype=x.dtype))
    threshold = jnp.maximum(requested, _constant_strain_series_threshold(x.dtype))
    use_series = x <= threshold**2
    x_safe = jnp.where(use_series, jnp.ones_like(x), x)
    return x, use_series, jnp.sqrt(x_safe)


def _adjoint_coefficients(
    angle_sq: Array, s: Array, eps: float | Array
) -> tuple[Array, Array, Array, Array]:
    r"""Return the four coefficients of the reduced adjoint exponential.

    With ``B = s * ad_xi`` and ``z = s * theta``, the exact reduced polynomial
    is

    .. math::

        \exp(B) = I + a_1(z)B + a_2(z)B^2
                    + a_3(z)B^3 + a_4(z)B^4.

    The regular coefficients are

    .. math::

        a_1 &= \frac{3\sin z-z\cos z}{2z}, &
        a_2 &= \frac{4-4\cos z-z\sin z}{2z^2}, \\
        a_3 &= \frac{\sin z-z\cos z}{2z^3}, &
        a_4 &= \frac{2-2\cos z-z\sin z}{2z^4}.

    Each removable quotient is replaced near zero by its even Taylor series
    through ``z**8``. Although ``z_safe`` is nonnegative, the result is valid
    for signed ``s`` because all four coefficients are even and the sign is
    carried by the powers of ``B``.

    Args:
        angle_sq: Squared rotational-strain magnitude.
        s: Scalar arclength used in ``B`` and ``z``.
        eps: Minimum rotational-strain series threshold.

    Returns:
        Four scalar arrays ``(a1, a2, a3, a4)`` in ascending matrix-power
        order.

    Notes:
        Strict-singularity mode disables the series and sanitization so the raw
        quotients remain observable at ``z == 0``.
    """
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
    r"""Return reduced tangent coefficients and their derivatives in ``x``.

    With ``B = s * ad_xi`` and ``x = (s * theta)**2``, the tangent operator is

    .. math::

        T(s,\xi) = s\left(I + c_1(x)B + c_2(x)B^2
                            + c_3(x)B^3 + c_4(x)B^4\right).

    This helper returns both ``c_k`` and the explicit analytic derivatives
    ``dc_k / dx``. The latter are consumed by the production implementation of
    ``D_xi T[xid]``; they are not generated with runtime autodiff. Both sets
    use Taylor polynomials through ``z**8`` in the small-angle branch and exact
    trigonometric quotients otherwise.

    Args:
        angle_sq: Squared rotational-strain magnitude.
        s: Scalar arclength held fixed by the tangent derivative.
        eps: Minimum rotational-strain series threshold.

    Returns:
        Pair ``(coefficients, derivatives)``. Each member is a four-tuple in
        ascending matrix-power order; ``derivatives[k]`` is
        ``d coefficients[k] / d x``.

    Notes:
        Differentiating with respect to ``x`` avoids the undefined expression
        ``dot(omega, omega_dot) / norm(omega)`` at zero spatial rotation. The
        caller supplies ``x_dot = s**2 * d(angle_sq)/dt`` instead.
    """
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
    """Build matrix powers through fourth order with shared products.

    Args:
        matrix: Square matrix ``B`` with shape ``(dim, dim)``.

    Returns:
        List ``[I, B, B**2, B**3, B**4]``. Every entry has the same shape and
        dtype as ``matrix``.

    Notes:
        Fourth order is exact for the reduced ``se(2)`` and ``se(3)`` adjoint
        polynomials; this is not a fourth-order approximation to a generic
        matrix exponential.
    """
    powers = [jnp.eye(matrix.shape[0], dtype=matrix.dtype)]
    for _ in range(4):
        powers.append(powers[-1] @ matrix)
    return powers


def _matrix_powers_with_derivatives(
    matrix: Array, matrix_dot: Array
) -> tuple[list[Array], list[Array]]:
    r"""Build fourth-order matrix powers and their directional derivatives.

    Starting with ``P_0 = I`` and ``Pdot_0 = 0``, the recurrence

    .. math::

        P_{k+1} &= P_k B, \\
        \dot P_{k+1} &= \dot P_k B + P_k \dot B

    applies the product rule without independently expanding every derivative.

    Args:
        matrix: Square matrix ``B`` with shape ``(dim, dim)``.
        matrix_dot: Directional derivative ``B_dot`` with the same shape.

    Returns:
        Pair ``(powers, dot_powers)``. The lists contain entries for orders zero
        through four, with ``dot_powers[k] = D(B**k)[B_dot]``.
    """
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
    """Evaluate the dimension-independent constant-strain adjoint polynomial.

    Args:
        ad_xi: ``se(2)`` or ``se(3)`` algebra-adjoint matrix with shape
            ``(dim, dim)``.
        angle_sq: Squared magnitude of the rotational part of ``xi``.
        s: Scalar arclength measured from the segment base.
        eps: Minimum rotational-strain series threshold.

    Returns:
        Matrix ``exp(s * ad_xi)`` with shape ``(dim, dim)``.

    Notes:
        The fourth-order reduced polynomial is algebraically exact for the
        supported Lie algebras. Only its scalar coefficients switch between
        Taylor and trigonometric representations.
    """
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
    r"""Evaluate the dimension-independent constant-strain tangent operator.

    This computes

    .. math::

        T(s,\xi) = \int_0^s \exp(\sigma\,\operatorname{ad}_\xi)\,d\sigma

    using the exact reduced fourth-order polynomial.

    Args:
        ad_xi: ``se(2)`` or ``se(3)`` algebra-adjoint matrix.
        angle_sq: Squared magnitude of the rotational strain.
        s: Scalar integration limit and segment arclength.
        eps: Minimum rotational-strain series threshold.

    Returns:
        Tangent matrix with the same shape and dtype as ``ad_xi``. At ``s == 0``
        the result is exactly the zero matrix in default mode.
    """
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
    r"""Evaluate the tangent's explicit directional derivative in ``xi``.

    The result is ``D_xi T(s, xi)[xid]`` with ``s`` held fixed. It combines the
    coefficient contribution
    ``(dc_k/dx) * x_dot * B**k`` and the power contribution
    ``c_k * D(B**k)[B_dot]``, where ``B_dot = s * ad_xid`` and
    ``x_dot = s**2 * angle_sq_dot``.

    Args:
        ad_xi: Algebra-adjoint matrix of the current strain.
        ad_xid: Algebra-adjoint matrix of the strain direction/rate.
        angle_sq: Squared rotational-strain magnitude.
        angle_sq_dot: Directional derivative of ``angle_sq``. It is
            ``2 * xi[0] * xid[0]`` for ``se(2)`` and
            ``2 * dot(omega, omega_dot)`` for ``se(3)``.
        s: Scalar arclength held fixed during differentiation.
        eps: Minimum rotational-strain series threshold.

    Returns:
        Directional derivative matrix with the same shape and dtype as
        ``ad_xi``.

    Notes:
        This is an analytic production path. JAX autodiff is used only by the
        test suite as an independent oracle.
    """
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
    r"""Return the adjoint accumulated along a constant planar strain segment.

    This is a rod-kinematics helper, not a generic transform adjoint. For a
    segment with constant planar strain ``xi`` evaluated at arclength ``s``,
    the returned matrix represents the closed-form adjoint associated with the
    segment transform ``exp(s * xi)`` under the angular-first ``se(2)``
    convention. Equivalently, the returned matrix is
    ``exp(s * se2.small_adjoint(xi))``. It transports planar twists between the
    material frames at the segment base and at arclength ``s``.

    Args:
        xi: Constant planar strain with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold. The actual dimensionless
            cutoff also accounts for ``s`` and the input dtype.

    Returns:
        Array with shape ``(3, 3)`` mapping planar twists through the segment
        adjoint at arclength ``s``.

    Notes:
        The implementation uses the exact reduced fourth-order polynomial, not
        a truncated matrix-exponential series. At zero curvature it retains
        the complete translation/shear dependence of ``exp(s * ad_xi)``. The
        Taylor branch affects only removable scalar coefficient quotients.
    """
    xi = jnp.asarray(xi).reshape(-1)
    angle_sq = xi[0] ** 2
    return _constant_strain_adjoint(se2.small_adjoint(xi), angle_sq, s, eps)


def adjoint_inverse_se2(xi: Array, s: Array, eps: float | Array) -> Array:
    """Return the inverse adjoint for a constant planar strain segment.

    This computes the inverse of :func:`adjoint_se2` without explicitly
    inverting the full homogeneous segment transform. It is used when
    propagating planar body-frame Jacobians backward through a constant-strain
    segment. For ``Ad = adjoint_se2(xi, s, eps)``, this function returns
    ``Ad**-1 = adjoint_se2(-xi, s, eps)`` while exploiting the block structure
    of an ``SE(2)`` adjoint instead of applying a generic matrix inverse.

    Args:
        xi: Constant planar strain with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold forwarded to
            :func:`adjoint_se2`.

    Returns:
        Array with shape ``(3, 3)`` representing the inverse segment adjoint.

    Notes:
        This path reuses the stabilized forward adjoint, so forward and inverse
        transport share one near-zero policy. It is intended for body-frame
        PCS/GVS recurrences rather than inversion of arbitrary matrices.
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
    r"""Return the tangent operator for a constant planar strain segment.

    The tangent operator integrates the local strain perturbation over
    arclength for a segment whose strain is constant. In PCS/GVS recurrences it
    maps strain-basis contributions into body-frame Jacobian contributions at
    arclength ``s``. Mathematically,

    .. math::

        T(s,\xi) = \int_0^s
        \exp(\sigma\,\operatorname{ad}_\xi)\,d\sigma.

    Args:
        xi: Constant planar strain with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold. The actual dimensionless
            cutoff also accounts for ``s`` and the input dtype.

    Returns:
        Array with shape ``(3, 3)`` containing the planar constant-strain
        tangent operator at arclength ``s``.

    Notes:
        The exact limits are ``T(0, xi) = 0`` and ``T(s, 0) = s I``. Pure
        translation is handled by the same reduced polynomial and does not
        require a separate curvature-zero formula.
    """
    xi = jnp.asarray(xi).reshape(-1)
    angle_sq = xi[0] ** 2
    return _constant_strain_tangent(se2.small_adjoint(xi), angle_sq, s, eps)


def tangent_derivative_se2(
    xi: Array, xid: Array, s: Array, eps: float | Array
) -> Array:
    r"""Return the directional derivative of the planar tangent operator.

    This differentiates :func:`tangent_se2` with respect to time through the
    strain ``xi`` and strain rate ``xid`` while holding arclength ``s`` fixed.
    The function uses the same angular-first planar convention as ``se2`` and
    computes ``D_xi T(s, xi)[xid]`` analytically.

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

    Notes:
        Both the reduced matrix powers and scalar coefficients are
        differentiated explicitly. No ``jvp``, ``grad``, or Jacobian transform
        is invoked in this production path. The result is linear in ``xid``
        and is exactly zero when either ``s == 0`` or ``xid == 0``.
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
    r"""Return the adjoint accumulated along a constant spatial strain segment.

    This is a rod-kinematics helper for spatial PCS/GVS segments. For constant
    strain ``xi`` evaluated at arclength ``s``, the returned matrix represents
    the closed-form adjoint associated with the segment transform
    ``exp(s * xi)`` under the angular-first ``se(3)`` convention.
    Equivalently, it is ``exp(s * se3.small_adjoint(xi))``.

    Args:
        xi: Constant spatial strain with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold. The actual dimensionless
            cutoff also accounts for ``s`` and the input dtype.

    Returns:
        Array with shape ``(6, 6)`` mapping spatial twists through the segment
        adjoint at arclength ``s``.

    Notes:
        Rotation enters the scalar coefficients only through
        ``dot(omega, omega)``. This avoids differentiating a Euclidean norm at
        zero while retaining arbitrary-axis behavior. The fourth-order matrix
        polynomial is exact for an ``se(3)`` adjoint.
    """
    xi = jnp.asarray(xi).reshape(-1)
    angle_sq = jnp.dot(xi[:3], xi[:3])
    return _constant_strain_adjoint(se3.small_adjoint(xi), angle_sq, s, eps)


def adjoint_inverse_se3(xi: Array, s: Array, eps: float | Array) -> Array:
    """Return the inverse adjoint for a constant spatial strain segment.

    This computes the inverse of :func:`adjoint_se3` without explicitly
    constructing and inverting the homogeneous segment transform. It is used
    when propagating body-frame Jacobians backward through spatial
    constant-strain segments. It uses the rotation and translation-skew blocks
    of the stabilized forward adjoint rather than a generic ``6 x 6`` inverse.

    Args:
        xi: Constant spatial strain with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold forwarded to
            :func:`adjoint_se3`.

    Returns:
        Array with shape ``(6, 6)`` representing the inverse segment adjoint.

    Notes:
        Building the inverse from the forward blocks keeps strict-singularity
        and stable-default behavior aligned with :func:`adjoint_se3` and avoids
        the cost and conditioning of a dense solve.
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
    r"""Return the tangent operator for a constant spatial strain segment.

    The tangent operator integrates local spatial strain perturbations over
    arclength for a segment whose strain is constant. In PCS/GVS recurrences it
    maps strain-basis contributions into body-frame Jacobian contributions at
    arclength ``s``. It evaluates

    .. math::

        T(s,\xi) = \int_0^s
        \exp(\sigma\,\operatorname{ad}_\xi)\,d\sigma

    with the exact reduced ``se(3)`` polynomial.

    Args:
        xi: Constant spatial strain with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold. The actual dimensionless
            cutoff also accounts for ``s`` and the input dtype.

    Returns:
        Array with shape ``(6, 6)`` containing the spatial constant-strain
        tangent operator at arclength ``s``.

    Notes:
        The implementation covers pure translation and arbitrary-axis rotation
        with one formula. At zero arclength the result is exactly zero; at zero
        strain it is ``s * I``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    angle_sq = jnp.dot(xi[:3], xi[:3])
    return _constant_strain_tangent(se3.small_adjoint(xi), angle_sq, s, eps)


def tangent_derivative_se3(
    xi: Array, xid: Array, s: Array, eps: float | Array
) -> Array:
    r"""Return the directional derivative of the spatial tangent operator.

    This differentiates :func:`tangent_se3` with respect to time through the
    strain ``xi`` and strain rate ``xid`` while holding arclength ``s`` fixed.
    The coefficient derivative is expressed through the squared rotational
    magnitude, avoiding a division by ``norm(omega)`` at zero rotation. The
    returned matrix is ``D_xi T(s, xi)[xid]``.

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

    Notes:
        The derivative uses ``d dot(omega, omega) / dt =
        2 * dot(omega, omega_dot)`` and the generic product-rule recurrence for
        powers of ``s * ad_xi``. It is fully analytic and linear in ``xid``;
        autodiff appears only in tests that validate this implementation.
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
