# ruff: noqa: I001, UP018
"""Runtime-sized cooperative GVS segment recurrence and assembly."""

from __future__ import annotations

import warp as wp

from soromox.systems.gvs._warp.lie import Vec6d

wp.set_module_options({"enable_backward": False})


SPATIAL_DIM = 6
BLOCK_DIM = 128


@wp.func
def _matrix_value(
    first: wp.array2d(dtype=wp.float64),
    second: wp.array2d(dtype=wp.float64),
    use_first: bool,
    base_row: int,
    row: int,
    column: int,
) -> wp.float64:
    if use_first:
        return first[base_row + row, column]
    return second[base_row + row, column]


@wp.func
def _vector_value(
    first: wp.array2d(dtype=wp.float64),
    second: wp.array2d(dtype=wp.float64),
    use_first: bool,
    base_row: int,
    row: int,
) -> wp.float64:
    if use_first:
        return first[base_row + row, 0]
    return second[base_row + row, 0]


@wp.func
def _write_matrix_value(
    first: wp.array2d(dtype=wp.float64),
    second: wp.array2d(dtype=wp.float64),
    use_first: bool,
    base_row: int,
    row: int,
    column: int,
    value: wp.float64,
):
    if use_first:
        first[base_row + row, column] = value
    else:
        second[base_row + row, column] = value


@wp.func
def _write_vector_value(
    first: wp.array2d(dtype=wp.float64),
    second: wp.array2d(dtype=wp.float64),
    use_first: bool,
    base_row: int,
    row: int,
    value: wp.float64,
):
    if use_first:
        first[base_row + row, 0] = value
    else:
        second[base_row + row, 0] = value


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
