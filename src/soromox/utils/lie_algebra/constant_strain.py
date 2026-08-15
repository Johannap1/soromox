r"""Closed-form Lie operators for constant-strain rod segments.

For a constant body strain ``xi`` and arclength ``s``, let
``A = ad_xi`` and ``B = s A``. This module evaluates three related operators:

.. math::

    \operatorname{Ad}(s, \xi) &= \exp(B), \\
    T(s, \xi) &= \int_0^s \exp(\sigma A)\,d\sigma, \\
    \dot T(s, \xi, \dot\xi)
        &= D_\xi T(s, \xi)[\dot\xi].

The minimal polynomials of ``se(2)`` and ``se(3)`` adjoint matrices reduce all
three expressions exactly to powers through ``B**4``. The spatial path uses
that reduced polynomial directly and can bundle related operators so PCS and
GVS evaluate the matrix powers and tangent coefficients only once. The planar
path further reduces the polynomial to exact ``2 x 2`` rotation/translation
blocks, avoiding generic ``3 x 3`` matrix powers. Public
``operators_se2``/``operators_se3`` bundles expose this sharing through a fixed
named result; private selective bundles let system recurrences omit outputs
they already have or do not consume.

The scalar trigonometric coefficients contain removable singularities at zero
rotation. Both paths therefore use closed forms away from zero and even Taylor
series through ``z**8`` near zero, where ``z = s * theta``. The coefficients
are expressed through ``x = z**2`` so neither the primal nor the explicit
tangent derivative divides by ``theta = norm(omega)``.

The default branch policy is designed to keep values, reverse-mode gradients,
and Hessians finite under scalar and batched JAX transformations. Under
``strict_singularities_mode`` the raw quotients are restored for diagnostics.
No automatic-differentiation operation is used in these production routines;
``tangent_derivative_se2`` and ``tangent_derivative_se3`` are explicit analytic
directional derivatives.
"""

__all__ = [
    "ConstantStrainOperators",
    "adjoint_se2",
    "adjoint_inverse_se2",
    "operators_se2",
    "tangent_se2",
    "tangent_derivative_se2",
    "adjoint_se3",
    "adjoint_inverse_se3",
    "operators_se3",
    "tangent_se3",
    "tangent_derivative_se3",
]

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from soromox.autodiff import strict_singularities_enabled

from . import se3
from ._jacobian_coefficients import (
    _forward_coefficients_and_x_derivatives,
    one_minus_cosine_over_angle_squared,
    sine_over_angle,
)


class ConstantStrainOperators(NamedTuple):
    """Related constant-strain Lie operators evaluated as one shared bundle.

    Attributes:
        adjoint: Forward adjoint ``exp(s * ad_xi)``.
        adjoint_inverse: Inverse of ``adjoint`` assembled from Lie-group
            blocks rather than a dense matrix inverse.
        tangent: Integrated adjoint
            ``integral_0^s exp(sigma * ad_xi) d sigma``.
        tangent_derivative: Analytic directional derivative
            ``D_xi tangent[xid]``. This is ``None`` when the bundle was
            requested without ``xid``.

    Notes:
        The named result gives the public bundled evaluators a fixed interface;
        unlike the private selective helpers, its field layout never depends
        on which values an internal system recurrence happens to consume.
    """

    adjoint: Array
    adjoint_inverse: Array
    tangent: Array
    tangent_derivative: Array | None


class _PreparedPowersSE3(NamedTuple):
    """Unscaled SE(3) adjoint powers shared across arclength evaluations.

    ``powers[k]`` contains ``ad_xi**k`` and ``dot_powers[k]`` its analytic
    directional derivative under ``ad_xid``. The scalar rotational invariants
    are stored alongside the matrices so an evaluator only needs to apply
    ``s**k`` and evaluate the stable coefficients at a requested arclength.
    """

    angle_sq: Array
    angle_sq_dot: Array
    powers: Array
    dot_powers: Array


# Shared reduced-polynomial coefficients and matrix-power recurrences.


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
    return _reduced_adjoint_from_powers(powers, angle_sq, s, eps)


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
    return _reduced_tangent_from_powers(powers, coefficients, s)


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
    return _reduced_tangent_derivative_from_powers(
        powers, dot_powers, coefficients, derivatives, angle_sq_dot, s
    )


def _reduced_adjoint_from_powers(
    powers: list[Array], angle_sq: Array, s: Array, eps: float | Array
) -> Array:
    r"""Assemble the exact reduced adjoint from precomputed powers of ``B``.

    ``powers`` must contain ``[I, B, ..., B**4]`` for
    ``B = s * ad_xi``. Supplying the powers lets a bundled evaluator reuse
    them for the tangent and its derivative. Only the scalar coefficients are
    selected here; this helper performs no additional matrix multiplication.
    """
    result = powers[0]
    for coefficient, power in zip(
        _adjoint_coefficients(angle_sq, s, eps), powers[1:], strict=True
    ):
        result = result + coefficient * power
    return result


def _reduced_tangent_from_powers(
    powers: list[Array],
    coefficients: tuple[Array, ...],
    s: Array,
) -> Array:
    r"""Assemble the exact reduced tangent from shared powers and coefficients.

    The result is ``s * (I + sum_k coefficients[k-1] * B**k)``. The caller is
    responsible for providing coefficients computed from the same ``xi`` and
    ``s`` as the powers, which allows ``T`` and ``Td`` to share both objects.
    """
    result = powers[0]
    for coefficient, power in zip(coefficients, powers[1:], strict=True):
        result = result + coefficient * power
    return s * result


def _reduced_transported_tangent_from_powers(
    powers: list[Array],
    coefficients: tuple[Array, ...],
    s: Array,
) -> Array:
    r"""Assemble ``exp(-B) @ T`` directly from shared powers.

    Since the tangent coefficients are even in ``z = s * theta``, changing
    ``B`` to ``-B`` only alternates the matrix-power signs. The identity

    .. math::

        \exp(-B)T(s,\xi) = -T(-s,\xi)
        = s\left(I-c_1B+c_2B^2-c_3B^3+c_4B^4\right)

    avoids a dense matrix product in PCS Jacobian recurrences while remaining
    exactly equivalent to the separately assembled operators.
    """
    result = powers[0]
    sign = -1.0
    for coefficient, power in zip(coefficients, powers[1:], strict=True):
        result = result + sign * coefficient * power
        sign = -sign
    return s * result


def _reduced_tangent_derivative_from_powers(
    powers: list[Array],
    dot_powers: list[Array],
    coefficients: tuple[Array, ...],
    derivatives: tuple[Array, ...],
    angle_sq_dot: Array,
    s: Array,
) -> Array:
    r"""Assemble the analytic tangent derivative from shared intermediates.

    Each term applies the product rule to its scalar coefficient and matrix
    power. ``derivatives`` are with respect to ``x = (s theta)**2`` and
    ``dot_powers[k]`` is ``D(B**k)[B_dot]``. Consequently no rotational norm
    or runtime autodiff transform is required at zero rotation.
    """
    x_dot = s**2 * angle_sq_dot
    result = jnp.zeros_like(powers[0])
    for coefficient, derivative, power, dot_power in zip(
        coefficients, derivatives, powers[1:], dot_powers[1:], strict=True
    ):
        result = result + derivative * x_dot * power + coefficient * dot_power
    return s * result


# Direct SE(2) block implementation and bundled evaluator.


def _forward_cutoff_se2(s: Array, eps: float | Array, dtype: jnp.dtype) -> Array:
    """Return the accumulated-angle cutoff for forward SE(2) coefficients.

    ``eps`` is expressed in rotational-strain units, whereas the scalar
    coefficients depend on ``z = s * theta``. Multiplication by ``abs(s)``
    converts the requested threshold to accumulated-angle units before it is
    combined with the dtype-aware ``u**(1/14)`` lower bound.
    """
    requested = jnp.abs(s) * jnp.abs(jnp.asarray(eps, dtype=dtype))
    return jnp.maximum(requested, _constant_strain_series_threshold(dtype))


def _forward_coefficients_se2(
    theta: Array, s: Array, eps: float | Array
) -> tuple[Array, tuple[Array, Array, Array]]:
    r"""Return ``z`` and the stable forward coefficient triple.

    The coefficients are ``sinc(z)``, ``cosc(z)``, and ``tanc(z)`` with
    ``z = s * theta``. They are evaluated together so their shared branch
    predicate and trigonometric work can be reused by the direct tangent-block
    assembler.
    """
    z = s * theta
    cutoff = _forward_cutoff_se2(s, eps, z.dtype)
    coefficients, _ = _forward_coefficients_and_x_derivatives(z, cutoff)
    return z, coefficients


def _adjoint_coefficients_se2(
    theta: Array, s: Array, eps: float | Array
) -> tuple[Array, Array, Array]:
    r"""Return ``z``, ``sinc(z)``, and ``cosc(z)`` for an SE(2) adjoint.

    The adjoint does not depend on ``tanc``. Avoiding its evaluation keeps the
    single-operator public adjoint functions smaller than the full bundled
    tangent path while retaining the same cutoff policy.
    """
    z = s * theta
    cutoff = _forward_cutoff_se2(s, eps, z.dtype)
    return (
        z,
        sine_over_angle(z, cutoff),
        one_minus_cosine_over_angle_squared(z, cutoff),
    )


def _forward_coefficients_and_derivatives_se2(
    theta: Array, s: Array, eps: float | Array
) -> tuple[
    Array,
    tuple[Array, Array, Array],
    tuple[Array, Array, Array],
]:
    r"""Return forward coefficients and their analytic ``z**2`` derivatives.

    The derivative triple is ``d(sinc, cosc, tanc) / d(z**2)``. Representing
    these even functions in ``z**2`` avoids dividing by ``theta`` at zero and
    lets :func:`_tangent_derivative_from_coefficients_se2` apply the chain rule
    using ``d(z**2)/dt = 2 z s xid[0]``.
    """
    z = s * theta
    cutoff = _forward_cutoff_se2(s, eps, z.dtype)
    coefficients, derivatives = _forward_coefficients_and_x_derivatives(z, cutoff)
    return z, coefficients, derivatives


def _adjoint_components_se2(
    xi: Array, s: Array, eps: float | Array
) -> tuple[Array, Array, Array]:
    r"""Evaluate the shared translation and rotation components of an adjoint.

    For ``xi = [theta, v]``, ``z = s * theta``, and
    ``a = -J v``, the lower blocks of the forward adjoint are

    .. math::

        q = s\left(\operatorname{sinc}(z)a
            + z\operatorname{cosc}(z)v\right),
        \qquad R = \exp(zJ).

    Returns:
        Tuple ``(q, sin_z, cos_z)``. These values are sufficient to assemble
        both the forward block ``[q, R]`` and inverse block ``[-R.T @ q, R.T]``
        without reevaluating a coefficient.

    Notes:
        This exact form replaces four generic ``3 x 3`` matrix products. The
        helper deliberately returns components rather than either matrix so a
        public operator bundle can construct both adjoints with shared work.
    """
    xi = jnp.asarray(xi).reshape(-1)
    s = jnp.asarray(s, dtype=xi.dtype)
    theta = xi[0]
    v = xi[1:]
    z, sinc, cosc = _adjoint_coefficients_se2(theta, s, eps)
    z_cosc = z * cosc
    minus_j_v = jnp.stack([v[1], -v[0]])
    q = s * (sinc * minus_j_v + z_cosc * v)

    return q, jnp.sin(z), jnp.cos(z)


def _adjoint_from_components_se2(
    q: Array, sin_z: Array, cos_z: Array, *, inverse: bool
) -> Array:
    r"""Assemble an SE(2) adjoint from precomputed scalar/block components.

    Args:
        q: Forward-adjoint translation column with shape ``(2,)``.
        sin_z: Sine of the signed accumulated rotation.
        cos_z: Cosine of the signed accumulated rotation.
        inverse: Select ``[-R.T @ q, R.T]`` instead of ``[q, R]``.

    Returns:
        Forward or inverse planar adjoint with shape ``(3, 3)``.
    """
    zero = jnp.zeros((), dtype=q.dtype)
    one = jnp.ones((), dtype=q.dtype)
    if inverse:
        q = -jnp.stack([cos_z * q[0] + sin_z * q[1], -sin_z * q[0] + cos_z * q[1]])
        return jnp.stack(
            [
                jnp.stack([one, zero, zero]),
                jnp.stack([q[0], cos_z, sin_z]),
                jnp.stack([q[1], -sin_z, cos_z]),
            ]
        )
    return jnp.stack(
        [
            jnp.stack([one, zero, zero]),
            jnp.stack([q[0], cos_z, -sin_z]),
            jnp.stack([q[1], sin_z, cos_z]),
        ]
    )


def _adjoint_block_se2(
    xi: Array, s: Array, eps: float | Array, *, inverse: bool
) -> Array:
    """Evaluate one planar adjoint while sharing its component assembly."""
    return _adjoint_from_components_se2(
        *_adjoint_components_se2(xi, s, eps), inverse=inverse
    )


def _tangent_from_coefficients_se2(
    xi: Array,
    s: Array,
    z: Array,
    coefficients: tuple[Array, Array, Array],
) -> Array:
    r"""Assemble the exact planar tangent from scalar coefficient blocks.

    For ``xi = [theta, v]`` and ``z = s * theta``, the lower-right block is
    ``s * (sinc(z) I + z cosc(z) J)`` and the lower-left column is
    ``s**2 * (cosc(z) (-J v) + z tanc(z) v)``. This block formula is exactly
    equal to the reduced matrix polynomial, including pure translation.
    """
    sinc, cosc, tanc = coefficients
    v = xi[1:]
    minus_j_v = jnp.stack([v[1], -v[0]])
    z_cosc = z * cosc
    z_tanc = z * tanc
    lower_left = s**2 * (cosc * minus_j_v + z_tanc * v)
    zero = jnp.zeros((), dtype=xi.dtype)
    return jnp.stack(
        [
            jnp.stack([s, zero, zero]),
            jnp.stack([lower_left[0], s * sinc, -s * z_cosc]),
            jnp.stack([lower_left[1], s * z_cosc, s * sinc]),
        ]
    )


def _tangent_derivative_from_coefficients_se2(
    xi: Array,
    xid: Array,
    s: Array,
    z: Array,
    coefficients: tuple[Array, Array, Array],
    derivatives: tuple[Array, Array, Array],
) -> Array:
    r"""Assemble ``D_xi T[xid]`` from analytic scalar derivatives.

    The chain rule is applied to the stable coefficient representation in
    ``x = z**2``. Both curvature-rate and translation-rate contributions are
    retained explicitly, so the result remains linear in ``xid`` and finite at
    zero curvature without invoking a JAX differentiation transform.
    """
    sinc, cosc, tanc = coefficients
    sinc_x, cosc_x, tanc_x = derivatives
    v = xi[1:]
    vd = xid[1:]
    minus_j_v = jnp.stack([v[1], -v[0]])
    minus_j_vd = jnp.stack([vd[1], -vd[0]])
    z_dot = s * xid[0]
    x = z**2
    sinc_dot = 2.0 * z * sinc_x * z_dot
    cosc_dot = 2.0 * z * cosc_x * z_dot
    z_cosc_dot = z_dot * (cosc + 2.0 * x * cosc_x)
    z_tanc = z * tanc
    z_tanc_dot = z_dot * (tanc + 2.0 * x * tanc_x)
    lower_left_dot = s**2 * (
        cosc_dot * minus_j_v + cosc * minus_j_vd + z_tanc_dot * v + z_tanc * vd
    )
    zero = jnp.zeros((), dtype=xi.dtype)
    return jnp.stack(
        [
            jnp.stack([zero, zero, zero]),
            jnp.stack([lower_left_dot[0], s * sinc_dot, -s * z_cosc_dot]),
            jnp.stack([lower_left_dot[1], s * z_cosc_dot, s * sinc_dot]),
        ]
    )


def _operators_se2(
    xi: Array,
    s: Array,
    adjoint_eps: float | Array,
    tangent_eps: float | Array,
    xid: Array | None = None,
) -> tuple[Array, Array] | tuple[Array, Array, Array]:
    r"""Evaluate an internal SE(2) operator bundle with shared scalar work.

    The returned operators are ``(Ad_inv, T)`` when ``xid`` is omitted and
    ``(Ad_inv, T, Td)`` otherwise. The latter path evaluates the tangent
    coefficients only once and reuses them for the tangent and its explicit
    directional derivative. Separate adjoint and tangent epsilon arguments
    preserve the threshold policies of internal system recurrences.

    This helper is private because it is an execution optimization for callers
    that need several public operators together; it is not a new Lie-algebra
    API. No automatic-differentiation transform is used in either path.

    Args:
        xi: Planar strain in angular-first order.
        s: Scalar arclength.
        adjoint_eps: Threshold for the inverse-adjoint coefficients.
        tangent_eps: Threshold for tangent coefficients and their derivatives.
        xid: Optional strain direction/rate. Supplying it requests ``Td``.

    Returns:
        ``(Ad_inv, T)`` or ``(Ad_inv, T, Td)``. Users who want a stable public
        named result should call :func:`operators_se2` instead.
    """
    xi = jnp.asarray(xi).reshape(-1)
    s = jnp.asarray(s, dtype=xi.dtype)
    adjoint_inverse = _adjoint_block_se2(xi, s, adjoint_eps, inverse=True)
    if xid is None:
        z, coefficients = _forward_coefficients_se2(xi[0], s, tangent_eps)
        return adjoint_inverse, _tangent_from_coefficients_se2(
            xi, s, z, coefficients
        )

    xid = jnp.asarray(xid).reshape(-1)
    z, coefficients, derivatives = _forward_coefficients_and_derivatives_se2(
        xi[0], s, tangent_eps
    )
    tangent = _tangent_from_coefficients_se2(xi, s, z, coefficients)
    tangent_derivative = _tangent_derivative_from_coefficients_se2(
        xi, xid, s, z, coefficients, derivatives
    )
    return adjoint_inverse, tangent, tangent_derivative


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
        The implementation uses the exact SE(2) rotation/translation block
        form, not a truncated matrix-exponential series. At zero curvature it
        retains the complete translation/shear dependence of
        ``exp(s * ad_xi)``. The Taylor branch affects only removable scalar
        coefficient quotients.
    """
    xi = jnp.asarray(xi).reshape(-1)
    return _adjoint_block_se2(xi, s, eps, inverse=False)


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
        This path shares the stabilized forward-adjoint block assembler, so
        forward and inverse transport have one near-zero policy. The inverse
        translation is formed explicitly as ``-R.T @ q`` without a generic
        matrix inverse. It is intended for body-frame PCS/GVS recurrences.
    """
    return _adjoint_block_se2(xi, s, eps, inverse=True)


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
    s = jnp.asarray(s, dtype=xi.dtype)
    z, coefficients = _forward_coefficients_se2(xi[0], s, eps)
    return _tangent_from_coefficients_se2(xi, s, z, coefficients)


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
        The SE(2) rotation/translation blocks and scalar coefficients are
        differentiated explicitly. No ``jvp``, ``grad``, or Jacobian transform
        is invoked in this production path. The result is linear in ``xid``
        and is exactly zero when either ``s == 0`` or ``xid == 0``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    xid = jnp.asarray(xid).reshape(-1)
    s = jnp.asarray(s, dtype=xi.dtype)
    z, coefficients, derivatives = _forward_coefficients_and_derivatives_se2(
        xi[0], s, eps
    )
    return _tangent_derivative_from_coefficients_se2(
        xi, xid, s, z, coefficients, derivatives
    )


def operators_se2(
    xi: Array,
    s: Array,
    eps: float | Array,
    xid: Array | None = None,
) -> ConstantStrainOperators:
    r"""Return the related SE(2) constant-strain operators as one bundle.

    Use this function when a caller needs more than one of
    :func:`adjoint_se2`, :func:`adjoint_inverse_se2`,
    :func:`tangent_se2`, and :func:`tangent_derivative_se2` at the same
    ``(xi, s)``. The bundled evaluation shares stabilized scalar coefficients
    and adjoint components instead of repeating them in separate public calls.

    Args:
        xi: Constant planar strain with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.
        s: Scalar arclength position measured from the segment base.
        eps: Minimum rotational-strain threshold used consistently by all
            returned operators.
        xid: Optional strain direction/rate with the same shape as ``xi``.
            When supplied, ``tangent_derivative`` contains
            ``D_xi T(s, xi)[xid]``; otherwise that field is ``None``.

    Returns:
        :class:`ConstantStrainOperators` containing the forward and inverse
        adjoints, tangent, and optional analytic tangent derivative.

    Notes:
        The result has a fixed named-field interface and is compatible with
        JAX transformations. No autodiff operation is executed internally.
        Call a single-operator function when only one expression is needed;
        this avoids assembling unused matrices.
    """
    xi = jnp.asarray(xi).reshape(-1)
    s = jnp.asarray(s, dtype=xi.dtype)
    adjoint_components = _adjoint_components_se2(xi, s, eps)
    adjoint = _adjoint_from_components_se2(*adjoint_components, inverse=False)
    adjoint_inverse = _adjoint_from_components_se2(
        *adjoint_components, inverse=True
    )
    if xid is None:
        z, coefficients = _forward_coefficients_se2(xi[0], s, eps)
        tangent = _tangent_from_coefficients_se2(xi, s, z, coefficients)
        tangent_derivative = None
    else:
        xid = jnp.asarray(xid).reshape(-1)
        z, coefficients, derivatives = _forward_coefficients_and_derivatives_se2(
            xi[0], s, eps
        )
        tangent = _tangent_from_coefficients_se2(xi, s, z, coefficients)
        tangent_derivative = _tangent_derivative_from_coefficients_se2(
            xi, xid, s, z, coefficients, derivatives
        )
    return ConstantStrainOperators(
        adjoint, adjoint_inverse, tangent, tangent_derivative
    )


# Reduced-polynomial SE(3) implementation and bundled evaluators.


def _adjoint_inverse_from_forward_se3(adjoint: Array) -> Array:
    r"""Construct an inverse SE(3) adjoint from stabilized forward blocks.

    For ``Ad = [[R, 0], [hat(t) R, R]]``, this assembles
    ``Ad^-1 = [[R.T, 0], [-R.T hat(t), R.T]]``. It avoids a dense inverse and
    guarantees that forward and inverse operators use identical stabilized
    coefficients.
    """
    rotation = adjoint[:3, :3]
    translation_hat_rotation = adjoint[3:, :3]
    rotation_inverse = jnp.transpose(rotation)
    translation_hat = translation_hat_rotation @ rotation_inverse
    zero = jnp.zeros((3, 3), dtype=adjoint.dtype)
    return jnp.block(
        [
            [rotation_inverse, zero],
            [-rotation_inverse @ translation_hat, rotation_inverse],
        ]
    )


def _prepare_powers_se3(xi: Array, xid: Array) -> _PreparedPowersSE3:
    r"""Precompute SE(3) powers that do not depend on arclength.

    PCS evaluates a constant segment strain at its tip and at several
    quadrature points. For ``A = ad_xi`` and fixed ``xid``, the identities

    .. math::

        (sA)^k = s^k A^k, \qquad
        D[(sA)^k][s\dot A] = s^k D[A^k][\dot A]

    let all matrix products be evaluated once per segment. This helper stores
    those unscaled powers; :func:`_operators_from_prepared_se3` applies the
    inexpensive scalar powers for each requested ``s``.

    Args:
        xi: Spatial strain in angular-first order.
        xid: Spatial strain direction/rate with the same shape.

    Returns:
        Prepared rotational invariants and stacked powers with shapes
        ``(5, 6, 6)``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    xid = jnp.asarray(xid).reshape(-1)
    powers, dot_powers = _matrix_powers_with_derivatives(
        se3.small_adjoint(xi), se3.small_adjoint(xid)
    )
    return _PreparedPowersSE3(
        angle_sq=jnp.dot(xi[:3], xi[:3]),
        angle_sq_dot=2.0 * jnp.dot(xi[:3], xid[:3]),
        powers=jnp.stack(powers),
        dot_powers=jnp.stack(dot_powers),
    )


def _scaled_powers_from_prepared_se3(
    prepared: _PreparedPowersSE3, s: Array
) -> tuple[list[Array], list[Array]]:
    """Scale prepared ``A**k`` and directional powers by ``s**k``."""
    scale = jnp.ones((), dtype=prepared.powers.dtype)
    powers: list[Array] = []
    dot_powers: list[Array] = []
    for order in range(5):
        powers.append(scale * prepared.powers[order])
        dot_powers.append(scale * prepared.dot_powers[order])
        scale = scale * s
    return powers, dot_powers


def _operators_from_powers_se3(
    powers: list[Array],
    dot_powers: list[Array] | None,
    angle_sq: Array,
    angle_sq_dot: Array | None,
    s: Array,
    adjoint_eps: float | Array,
    tangent_eps: float | Array,
    *,
    return_adjoint: bool = False,
) -> (
    tuple[Array, Array]
    | tuple[Array, Array, Array]
    | tuple[Array, Array, Array, Array]
):
    """Assemble selected SE(3) operators from already scaled powers."""
    adjoint = _reduced_adjoint_from_powers(powers, angle_sq, s, adjoint_eps)
    adjoint_inverse = _adjoint_inverse_from_forward_se3(adjoint)
    coefficients, derivatives = _tangent_coefficients_and_x_derivatives(
        angle_sq, s, tangent_eps
    )
    tangent = _reduced_tangent_from_powers(powers, coefficients, s)

    outputs = (adjoint_inverse, tangent)
    if dot_powers is not None:
        assert angle_sq_dot is not None
        tangent_derivative = _reduced_tangent_derivative_from_powers(
            powers,
            dot_powers,
            coefficients,
            derivatives,
            angle_sq_dot,
            s,
        )
        outputs += (tangent_derivative,)
    if return_adjoint:
        outputs += (adjoint,)
    return outputs


def _operators_from_prepared_se3(
    prepared: _PreparedPowersSE3,
    s: Array,
    adjoint_eps: float | Array,
    tangent_eps: float | Array,
) -> tuple[Array, Array, Array]:
    """Evaluate ``(Ad_inv, T, Td)`` from per-segment prepared powers."""
    s = jnp.asarray(s, dtype=prepared.powers.dtype)
    powers, dot_powers = _scaled_powers_from_prepared_se3(prepared, s)
    operators = _operators_from_powers_se3(
        powers,
        dot_powers,
        prepared.angle_sq,
        prepared.angle_sq_dot,
        s,
        adjoint_eps,
        tangent_eps,
    )
    assert len(operators) == 3
    return operators


def _kinematic_operators_from_prepared_se3(
    prepared: _PreparedPowersSE3,
    s: Array,
    adjoint_eps: float | Array,
    tangent_eps: float | Array,
    *,
    convective_only: bool,
) -> tuple[Array, Array, Array] | tuple[Array, Array, Array, Array]:
    r"""Evaluate the operator subset consumed by PCS recurrences.

    Both paths return ``Ad_inv`` and the directly assembled transported
    tangent ``Ad_inv @ T``. The convective path returns ``Td`` as its third
    value and deliberately omits ``T``; forward dynamics only needs
    ``Ad_inv @ (Td @ xid)``. The full-derivative path returns
    ``(Ad_inv, Ad_inv_T, T, Td)``.

    This selective helper keeps the public operator bundle fixed while
    avoiding matrices that an internal recurrence would immediately discard.
    """
    s = jnp.asarray(s, dtype=prepared.powers.dtype)
    powers, dot_powers = _scaled_powers_from_prepared_se3(prepared, s)
    adjoint = _reduced_adjoint_from_powers(
        powers, prepared.angle_sq, s, adjoint_eps
    )
    adjoint_inverse = _adjoint_inverse_from_forward_se3(adjoint)
    coefficients, derivatives = _tangent_coefficients_and_x_derivatives(
        prepared.angle_sq, s, tangent_eps
    )
    transported_tangent = _reduced_transported_tangent_from_powers(
        powers, coefficients, s
    )
    tangent_derivative = _reduced_tangent_derivative_from_powers(
        powers,
        dot_powers,
        coefficients,
        derivatives,
        prepared.angle_sq_dot,
        s,
    )
    if convective_only:
        return adjoint_inverse, transported_tangent, tangent_derivative
    tangent = _reduced_tangent_from_powers(powers, coefficients, s)
    return adjoint_inverse, transported_tangent, tangent, tangent_derivative


def _operators_se3(
    xi: Array,
    s: Array,
    adjoint_eps: float | Array,
    tangent_eps: float | Array,
    xid: Array | None = None,
    *,
    return_adjoint: bool = False,
) -> (
    tuple[Array, Array]
    | tuple[Array, Array, Array]
    | tuple[Array, Array, Array, Array]
):
    r"""Evaluate related SE(3) constant-strain operators with shared work.

    The standard result is ``(Ad_inv, T)``. Supplying ``xid`` adds ``Td``;
    requesting ``return_adjoint`` adds the forward adjoint as the final
    element. Matrix powers through ``B**4`` are built once, while ``T`` and
    ``Td`` also share their scalar coefficients. Separate adjoint and tangent
    thresholds preserve the caller's existing numerical policy.

    This private execution helper mirrors :func:`_operators_se2` while keeping
    the spatial fourth-order polynomial. It performs no autodiff operation.

    Args:
        xi: Spatial strain in angular-first order.
        s: Scalar arclength.
        adjoint_eps: Threshold for forward/inverse adjoint coefficients.
        tangent_eps: Threshold for tangent coefficients and derivatives.
        xid: Optional strain direction/rate. Supplying it requests ``Td``.
        return_adjoint: Append the already-computed forward adjoint. This is
            used by arc-length derivatives and the public named bundle.

    Returns:
        Selective tuple beginning with ``(Ad_inv, T)`` and followed by ``Td``
        and/or ``Ad`` according to the two optional arguments. The public
        :func:`operators_se3` function provides a fixed named return type.
    """
    xi = jnp.asarray(xi).reshape(-1)
    s = jnp.asarray(s, dtype=xi.dtype)
    ad_xi = se3.small_adjoint(xi)
    angle_sq = jnp.dot(xi[:3], xi[:3])

    if xid is None:
        powers = _matrix_powers(s * ad_xi)
        dot_powers = None
    else:
        xid = jnp.asarray(xid).reshape(-1)
        powers, dot_powers = _matrix_powers_with_derivatives(
            s * ad_xi, s * se3.small_adjoint(xid)
        )

    angle_sq_dot = None if xid is None else 2.0 * jnp.dot(xi[:3], xid[:3])
    return _operators_from_powers_se3(
        powers,
        dot_powers,
        angle_sq,
        angle_sq_dot,
        s,
        adjoint_eps,
        tangent_eps,
        return_adjoint=return_adjoint,
    )


def _tangent_and_derivative_se3(
    xi: Array,
    xid: Array,
    s: Array,
    eps: float | Array,
) -> tuple[Array, Array]:
    r"""Evaluate spatial ``T`` and ``Td`` while sharing powers and coefficients.

    This is the selective GVS counterpart to :func:`_operators_se3`: it does
    not construct an unused adjoint because GVS already obtains that operator
    from the relative pose required by its kinematic recurrence.

    Args:
        xi: Spatial strain in angular-first order.
        xid: Strain direction/rate.
        s: Scalar arclength held fixed during differentiation.
        eps: Tangent coefficient threshold.

    Returns:
        Tuple ``(T, Td)`` with both arrays having shape ``(6, 6)``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    xid = jnp.asarray(xid).reshape(-1)
    s = jnp.asarray(s, dtype=xi.dtype)
    ad_xi = se3.small_adjoint(xi)
    powers, dot_powers = _matrix_powers_with_derivatives(
        s * ad_xi, s * se3.small_adjoint(xid)
    )
    angle_sq = jnp.dot(xi[:3], xi[:3])
    angle_sq_dot = 2.0 * jnp.dot(xi[:3], xid[:3])
    coefficients, derivatives = _tangent_coefficients_and_x_derivatives(
        angle_sq, s, eps
    )
    tangent = _reduced_tangent_from_powers(powers, coefficients, s)
    tangent_derivative = _reduced_tangent_derivative_from_powers(
        powers,
        dot_powers,
        coefficients,
        derivatives,
        angle_sq_dot,
        s,
    )
    return tangent, tangent_derivative


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
    adjoint = adjoint_se3(xi, s, eps=eps)
    return _adjoint_inverse_from_forward_se3(adjoint)


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


def operators_se3(
    xi: Array,
    s: Array,
    eps: float | Array,
    xid: Array | None = None,
) -> ConstantStrainOperators:
    r"""Return the related SE(3) constant-strain operators as one bundle.

    This is the spatial counterpart of :func:`operators_se2`. It is intended
    for users who require multiple expressions at the same constant strain and
    arclength. Matrix powers through ``B**4`` are constructed once; the
    tangent and its optional derivative also share their stable coefficients.

    Args:
        xi: Constant spatial strain with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        s: Scalar arclength position measured from the segment base.
        eps: Minimum rotational-strain threshold used consistently by all
            returned operators.
        xid: Optional strain direction/rate with the same shape as ``xi``.
            When supplied, ``tangent_derivative`` contains
            ``D_xi T(s, xi)[xid]``; otherwise that field is ``None``.

    Returns:
        :class:`ConstantStrainOperators` containing the forward and inverse
        adjoints, tangent, and optional analytic tangent derivative.

    Notes:
        The fixed named result is JAX-transformable. The implementation uses
        explicit matrix-power and coefficient derivatives and performs no
        runtime autodiff. For one expression, prefer the corresponding
        single-operator function to avoid unused assembly.
    """
    if xid is None:
        adjoint_inverse, tangent, adjoint = _operators_se3(
            xi, s, eps, eps, return_adjoint=True
        )
        tangent_derivative = None
    else:
        adjoint_inverse, tangent, tangent_derivative, adjoint = _operators_se3(
            xi, s, eps, eps, xid, return_adjoint=True
        )
    return ConstantStrainOperators(
        adjoint, adjoint_inverse, tangent, tangent_derivative
    )
