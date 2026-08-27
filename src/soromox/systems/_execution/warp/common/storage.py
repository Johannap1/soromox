# ruff: noqa: I001, UP018
"""Double-buffer access helpers shared by persistent Warp kernels."""

from __future__ import annotations

import warp as wp


wp.set_module_options({"enable_backward": False})


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
