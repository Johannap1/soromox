# ruff: noqa: I001, UP018
"""Shared Warp operators for planar rigid motions."""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.func
def _forward_coefficients(z: wp.float64, cutoff: wp.float64) -> wp.vec3d:
    """Evaluate stable scalar coefficients for planar exponential operators.

    Args:
        z: Integrated planar rotation.
        cutoff: Magnitude below which a polynomial expansion is used.

    Returns:
        Stable ``(sinc, cosc, tanc)`` coefficients.
    """
    x = z * z
    if wp.abs(z) <= cutoff:
        return wp.vec3d(
            wp.float64(1.0)
            + x
            * (
                -wp.float64(1.0 / 6.0)
                + x
                * (
                    wp.float64(1.0 / 120.0)
                    + x * (-wp.float64(1.0 / 5040.0) + x * wp.float64(1.0 / 362880.0))
                )
            ),
            wp.float64(0.5)
            + x
            * (
                -wp.float64(1.0 / 24.0)
                + x
                * (
                    wp.float64(1.0 / 720.0)
                    + x * (-wp.float64(1.0 / 40320.0) + x * wp.float64(1.0 / 3628800.0))
                )
            ),
            wp.float64(1.0 / 6.0)
            + x
            * (
                -wp.float64(1.0 / 120.0)
                + x
                * (
                    wp.float64(1.0 / 5040.0)
                    + x
                    * (-wp.float64(1.0 / 362880.0) + x * wp.float64(1.0 / 39916800.0))
                )
            ),
        )
    sine = wp.sin(z)
    cosine = wp.cos(z)
    return wp.vec3d(
        sine / z,
        (wp.float64(1.0) - cosine) / x,
        (z - sine) / (x * z),
    )


@wp.func
def _forward_coefficient_x_derivatives(z: wp.float64, cutoff: wp.float64) -> wp.vec3d:
    """Differentiate planar exponential coefficients with respect to ``z**2``.

    Args:
        z: Integrated planar rotation.
        cutoff: Magnitude below which a polynomial expansion is used.

    Returns:
        Derivatives of ``(sinc, cosc, tanc)`` with respect to ``z**2``.
    """
    x = z * z
    if wp.abs(z) <= cutoff:
        return wp.vec3d(
            -wp.float64(1.0 / 6.0)
            + x
            * (
                wp.float64(1.0 / 60.0)
                + x * (-wp.float64(1.0 / 1680.0) + x * wp.float64(1.0 / 90720.0))
            ),
            -wp.float64(1.0 / 24.0)
            + x
            * (
                wp.float64(1.0 / 360.0)
                + x * (-wp.float64(1.0 / 13440.0) + x * wp.float64(1.0 / 907200.0))
            ),
            -wp.float64(1.0 / 120.0)
            + x
            * (
                wp.float64(1.0 / 2520.0)
                + x * (-wp.float64(1.0 / 120960.0) + x * wp.float64(1.0 / 9979200.0))
            ),
        )
    sine = wp.sin(z)
    cosine = wp.cos(z)
    return wp.vec3d(
        (z * cosine - sine) / (wp.float64(2.0) * x * z),
        (z * sine + wp.float64(2.0) * cosine - wp.float64(2.0))
        / (wp.float64(2.0) * x * x),
        (wp.float64(3.0) * sine - wp.float64(2.0) * z - z * cosine)
        / (wp.float64(2.0) * x * x * z),
    )


@wp.func
def _constant_strain_operators(
    xi: wp.vec3d,
    xid: wp.vec3d,
    s: wp.float64,
    adjoint_eps: wp.float64,
    tangent_eps: wp.float64,
) -> tuple[wp.mat33d, wp.mat33d, wp.vec3d, wp.vec3d]:
    """Evaluate constant-strain SE(2) recurrence operators.

    Args:
        xi: Constant planar strain of the current segment.
        xid: Constant planar strain rate of the current segment.
        s: Segment-local integration coordinate.
        adjoint_eps: Small-angle tolerance for the exponential adjoint.
        tangent_eps: Small-angle tolerance for tangent derivatives.

    Returns:
        Inverse adjoint, transported left tangent, local velocity, and the
        transported tangent time-derivative action on strain rate.
    """
    z = s * xi[0]
    adjoint_cutoff = wp.max(wp.abs(s * adjoint_eps), wp.float64(0.04964607461902946))
    adjoint_coefficients = _forward_coefficients(z, adjoint_cutoff)
    adjoint_sinc = adjoint_coefficients[0]
    adjoint_cosc = adjoint_coefficients[1]
    cosine = wp.cos(z)
    sine = wp.sin(z)
    adjoint_translation = wp.vec2d(
        s * (adjoint_sinc * xi[2] + z * adjoint_cosc * xi[1]),
        s * (-adjoint_sinc * xi[1] + z * adjoint_cosc * xi[2]),
    )

    adjoint_inverse = wp.mat33d()
    adjoint_inverse[0, 0] = wp.float64(1.0)
    adjoint_inverse[1, 0] = -(
        cosine * adjoint_translation[0] + sine * adjoint_translation[1]
    )
    adjoint_inverse[1, 1] = cosine
    adjoint_inverse[1, 2] = sine
    adjoint_inverse[2, 0] = (
        sine * adjoint_translation[0] - cosine * adjoint_translation[1]
    )
    adjoint_inverse[2, 1] = -sine
    adjoint_inverse[2, 2] = cosine

    tangent_cutoff = wp.max(wp.abs(s * tangent_eps), wp.float64(0.07618835359095202))
    coefficients = _forward_coefficients(z, tangent_cutoff)
    coefficient_x_derivatives = _forward_coefficient_x_derivatives(z, tangent_cutoff)
    sinc = coefficients[0]
    cosc = coefficients[1]
    tanc = coefficients[2]
    accumulated_x = s * xi[1]
    accumulated_y = s * xi[2]
    lower_left = wp.vec2d(
        cosc * accumulated_y + z * tanc * accumulated_x,
        -cosc * accumulated_x + z * tanc * accumulated_y,
    )
    tangent = wp.mat33d()
    tangent[0, 0] = s
    tangent[1, 0] = s * lower_left[0]
    tangent[1, 1] = s * sinc
    tangent[1, 2] = -s * z * cosc
    tangent[2, 0] = s * lower_left[1]
    tangent[2, 1] = s * z * cosc
    tangent[2, 2] = s * sinc

    accumulated_dot_x = s * xid[1]
    accumulated_dot_y = s * xid[2]
    z_dot = s * xid[0]
    x = z * z
    sinc_x = coefficient_x_derivatives[0]
    cosc_x = coefficient_x_derivatives[1]
    tanc_x = coefficient_x_derivatives[2]
    sinc_dot = wp.float64(2.0) * z * sinc_x * z_dot
    cosc_dot = wp.float64(2.0) * z * cosc_x * z_dot
    z_cosc_dot = z_dot * (cosc + wp.float64(2.0) * x * cosc_x)
    z_tanc = z * tanc
    z_tanc_dot = z_dot * (tanc + wp.float64(2.0) * x * tanc_x)
    lower_left_dot = wp.vec2d(
        cosc_dot * accumulated_y
        + cosc * accumulated_dot_y
        + z_tanc_dot * accumulated_x
        + z_tanc * accumulated_dot_x,
        -cosc_dot * accumulated_x
        - cosc * accumulated_dot_x
        + z_tanc_dot * accumulated_y
        + z_tanc * accumulated_dot_y,
    )
    tangent_dot = wp.mat33d()
    tangent_dot[1, 0] = s * lower_left_dot[0]
    tangent_dot[1, 1] = s * sinc_dot
    tangent_dot[1, 2] = -s * z_cosc_dot
    tangent_dot[2, 0] = s * lower_left_dot[1]
    tangent_dot[2, 1] = s * z_cosc_dot
    tangent_dot[2, 2] = s * sinc_dot

    transported_tangent = adjoint_inverse * tangent
    local_velocity = transported_tangent * xid
    transported_tangent_dot_velocity = adjoint_inverse * (tangent_dot * xid)
    return (
        adjoint_inverse,
        transported_tangent,
        local_velocity,
        transported_tangent_dot_velocity,
    )


@wp.func
def planar_pose_step(
    theta: wp.float64,
    x: wp.float64,
    y: wp.float64,
    xi: wp.vec3d,
    arc_length: wp.float64,
    eps: wp.float64,
) -> wp.vec3d:
    """Integrate one constant-strain SE(2) pose step.

    Args:
        theta: Absolute orientation at the start of the step.
        x: Absolute horizontal position at the start of the step.
        y: Absolute vertical position at the start of the step.
        xi: Constant planar strain in angular-first coordinates.
        arc_length: Local integration distance.
        eps: Requested small-angle threshold.

    Returns:
        Absolute planar pose ``[theta, x, y]`` after the step.
    """
    z = arc_length * xi[0]
    cutoff = wp.max(wp.abs(arc_length * eps), wp.float64(0.04964607461902946))
    coefficients = _forward_coefficients(z, cutoff)
    vx = arc_length * xi[1]
    vy = arc_length * xi[2]
    local_x = coefficients[0] * vx - z * coefficients[1] * vy
    local_y = z * coefficients[1] * vx + coefficients[0] * vy
    cosine = wp.cos(theta)
    sine = wp.sin(theta)
    return wp.vec3d(
        theta + z,
        x + cosine * local_x - sine * local_y,
        y + sine * local_x + cosine * local_y,
    )


__all__ = [
    "_constant_strain_operators",
    "_forward_coefficients",
    "_forward_coefficient_x_derivatives",
    "planar_pose_step",
]
