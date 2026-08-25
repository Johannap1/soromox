# ruff: noqa: I001, UP018
"""Forward-only runtime-sized batched dense Cholesky solve for GVS."""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})


BLOCK_DIM = 128


@wp.kernel(enable_backward=False)
def scalable_cholesky_solve_kernel(
    inertia: wp.array3d(dtype=wp.float64),
    coriolis_qd: wp.array2d(dtype=wp.float64),
    gravity_force: wp.array2d(dtype=wp.float64),
    lanes_per_block: wp.array(dtype=wp.int32),
    factor: wp.array3d(dtype=wp.float64),
    intermediate: wp.array2d(dtype=wp.float64),
    acceleration: wp.array2d(dtype=wp.float64),
):
    """Factor and solve one SPD system per cooperative block."""

    environment, lane = wp.tid()
    num_dofs = inertia.shape[1]
    lane_stride = lanes_per_block[0]
    barrier = wp.tile_zeros(shape=(1,), dtype=wp.int32, storage="shared")

    entry = lane
    while entry < num_dofs * num_dofs:
        row = entry // num_dofs
        column = entry - row * num_dofs
        if row >= column:
            factor[environment, row, column] = inertia[
                environment, row, column
            ]
        entry += lane_stride
    column = lane
    while column < num_dofs:
        intermediate[environment, column] = -(
            coriolis_qd[environment, column]
            + gravity_force[environment, column]
        )
        acceleration[environment, column] = wp.float64(0.0)
        column += lane_stride
    wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)

    pivot = int(0)
    while pivot < num_dofs:
        if lane == 0:
            diagonal = factor[environment, pivot, pivot]
            k = int(0)
            while k < pivot:
                value = factor[environment, pivot, k]
                diagonal -= value * value
                k += 1
            factor[environment, pivot, pivot] = wp.sqrt(diagonal)
        wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)

        row = pivot + 1 + lane
        while row < num_dofs:
            value = factor[environment, row, pivot]
            k = int(0)
            while k < pivot:
                value -= (
                    factor[environment, row, k]
                    * factor[environment, pivot, k]
                )
                k += 1
            factor[environment, row, pivot] = (
                value / factor[environment, pivot, pivot]
            )
            row += lane_stride
        wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
        pivot += 1

    row = int(0)
    while row < num_dofs:
        if lane == 0:
            value = intermediate[environment, row]
            k = int(0)
            while k < row:
                value -= (
                    factor[environment, row, k]
                    * intermediate[environment, k]
                )
                k += 1
            intermediate[environment, row] = (
                value / factor[environment, row, row]
            )
        wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
        row += 1

    row = num_dofs - 1
    while row >= 0:
        if lane == 0:
            value = intermediate[environment, row]
            k = row + 1
            while k < num_dofs:
                value -= (
                    factor[environment, k, row]
                    * acceleration[environment, k]
                )
                k += 1
            acceleration[environment, row] = (
                value / factor[environment, row, row]
            )
        wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
        row -= 1


def _scalable_cholesky_solve(
    inertia: wp.array3d(dtype=wp.float64),
    coriolis_qd: wp.array2d(dtype=wp.float64),
    gravity_force: wp.array2d(dtype=wp.float64),
    lanes_per_block: wp.array(dtype=wp.int32),
    factor: wp.array3d(dtype=wp.float64),
    intermediate: wp.array2d(dtype=wp.float64),
    acceleration: wp.array2d(dtype=wp.float64),
):
    wp.launch_tiled(
        scalable_cholesky_solve_kernel,
        dim=inertia.shape[0],
        inputs=[inertia, coriolis_qd, gravity_force, lanes_per_block],
        outputs=[factor, intermediate, acceleration],
        block_dim=BLOCK_DIM,
    )


scalable_cholesky_solve = wp.jax_callable(
    _scalable_cholesky_solve, num_outputs=3
)
