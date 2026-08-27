# ruff: noqa: I001, UP018
"""Spatial inertia and force helpers shared by SE(3) Warp kernels."""

from __future__ import annotations

import warp as wp

from soromox.systems._execution.warp.common.se3 import Vec6d
from soromox.systems._execution.warp.common.storage import _vector_value

wp.set_module_options({"enable_backward": False})


@wp.func
def _coadjoint_wrench(
    velocity_first: wp.array2d(dtype=wp.float64),
    velocity_second: wp.array2d(dtype=wp.float64),
    derivative_first: wp.array2d(dtype=wp.float64),
    derivative_second: wp.array2d(dtype=wp.float64),
    use_first: bool,
    base_row: int,
    masses: wp.array2d(dtype=wp.float64),
    quadrature: int,
) -> Vec6d:
    omega = wp.vec3d(
        _vector_value(
            velocity_first, velocity_second, use_first, base_row, 0
        ),
        _vector_value(
            velocity_first, velocity_second, use_first, base_row, 1
        ),
        _vector_value(
            velocity_first, velocity_second, use_first, base_row, 2
        ),
    )
    linear_velocity = wp.vec3d(
        _vector_value(
            velocity_first, velocity_second, use_first, base_row, 3
        ),
        _vector_value(
            velocity_first, velocity_second, use_first, base_row, 4
        ),
        _vector_value(
            velocity_first, velocity_second, use_first, base_row, 5
        ),
    )
    moment = wp.vec3d(
        masses[quadrature, 0] * omega[0],
        masses[quadrature, 1] * omega[1],
        masses[quadrature, 2] * omega[2],
    )
    force = wp.vec3d(
        masses[quadrature, 3] * linear_velocity[0],
        masses[quadrature, 4] * linear_velocity[1],
        masses[quadrature, 5] * linear_velocity[2],
    )
    angular = wp.cross(omega, moment) + wp.cross(linear_velocity, force)
    linear = wp.cross(omega, force)
    return Vec6d(
        masses[quadrature, 0]
        * _vector_value(
            derivative_first, derivative_second, use_first, base_row, 0
        )
        + angular[0],
        masses[quadrature, 1]
        * _vector_value(
            derivative_first, derivative_second, use_first, base_row, 1
        )
        + angular[1],
        masses[quadrature, 2]
        * _vector_value(
            derivative_first, derivative_second, use_first, base_row, 2
        )
        + angular[2],
        masses[quadrature, 3]
        * _vector_value(
            derivative_first, derivative_second, use_first, base_row, 3
        )
        + linear[0],
        masses[quadrature, 4]
        * _vector_value(
            derivative_first, derivative_second, use_first, base_row, 4
        )
        + linear[1],
        masses[quadrature, 5]
        * _vector_value(
            derivative_first, derivative_second, use_first, base_row, 5
        )
        + linear[2],
    )
