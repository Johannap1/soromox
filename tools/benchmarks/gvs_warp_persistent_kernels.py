# ruff: noqa: I001, UP018
"""Runtime-sized persistent whole-chain GVS dynamics kernel."""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})

from tools.benchmarks.gvs_warp_lie_kernels import Vec6d, _ad_action
from tools.benchmarks.gvs_warp_segment_kernels import (
    _coadjoint_wrench,
    _matrix_value,
    _vector_value,
    _write_matrix_value,
    _write_vector_value,
)


SPATIAL_DIM = 6
BLOCK_DIM = 128


@wp.kernel(enable_backward=False)
def scalable_persistent_chain_kernel(
    joint_adjoint: wp.array2d(dtype=wp.float64),
    joint_adjoint_dot: wp.array2d(dtype=wp.float64),
    joint_tangent: wp.array2d(dtype=wp.float64),
    joint_tangent_dot_qd: wp.array2d(dtype=wp.float64),
    joint_velocity: wp.array2d(dtype=wp.float64),
    cell_adjoint: wp.array2d(dtype=wp.float64),
    cell_tangent_local: wp.array2d(dtype=wp.float64),
    cell_link_velocity: wp.array2d(dtype=wp.float64),
    cell_step_velocity: wp.array2d(dtype=wp.float64),
    cell_tangent_velocity_dot: wp.array2d(dtype=wp.float64),
    global_to_local: wp.array2d(dtype=wp.int32),
    active_dofs: wp.array(dtype=wp.int32),
    qd: wp.array2d(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    masses: wp.array2d(dtype=wp.float64),
    gravity_base: wp.array(dtype=wp.float64),
    num_cells_array: wp.array(dtype=wp.int32),
    num_quadrature_array: wp.array(dtype=wp.int32),
    lanes_per_block: wp.array(dtype=wp.int32),
    jacobian_first: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_first: wp.array2d(dtype=wp.float64),
    velocity_first: wp.array2d(dtype=wp.float64),
    gravity_first: wp.array2d(dtype=wp.float64),
    jacobian_second: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_second: wp.array2d(dtype=wp.float64),
    velocity_second: wp.array2d(dtype=wp.float64),
    gravity_second: wp.array2d(dtype=wp.float64),
    inertia: wp.array3d(dtype=wp.float64),
    coriolis_qd: wp.array2d(dtype=wp.float64),
    gravity_force: wp.array2d(dtype=wp.float64),
):
    """Traverse one full serial GVS chain per cooperative block."""

    environment, lane = wp.tid()
    num_dofs = qd.shape[1]
    num_cells = num_cells_array[0]
    num_quadrature = num_quadrature_array[0]
    num_segments = joint_adjoint.shape[0] // (qd.shape[0] * SPATIAL_DIM)
    lane_stride = lanes_per_block[0]
    state_base_row = environment * SPATIAL_DIM
    barrier = wp.tile_zeros(shape=(1,), dtype=wp.int32, storage="shared")

    entry = lane
    while entry < SPATIAL_DIM * num_dofs:
        row = entry // num_dofs
        column = entry - row * num_dofs
        jacobian_first[state_base_row + row, column] = wp.float64(0.0)
        jacobian_second[state_base_row + row, column] = wp.float64(0.0)
        entry += lane_stride
    row = lane
    while row < SPATIAL_DIM:
        jacobian_dot_qd_first[state_base_row + row, 0] = wp.float64(0.0)
        velocity_first[state_base_row + row, 0] = wp.float64(0.0)
        gravity_first[state_base_row + row, 0] = gravity_base[row]
        row += lane_stride
    entry = lane
    while entry < num_dofs * num_dofs:
        output_row = entry // num_dofs
        output_column = entry - output_row * num_dofs
        inertia[environment, output_row, output_column] = wp.float64(0.0)
        entry += lane_stride
    column = lane
    while column < num_dofs:
        coriolis_qd[environment, column] = wp.float64(0.0)
        gravity_force[environment, column] = wp.float64(0.0)
        column += lane_stride
    wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)

    current_is_first = bool(True)
    segment = int(0)
    while segment < num_segments:
        segment_dofs = active_dofs[segment]
        joint_base_row = (
            (environment * num_segments + segment) * SPATIAL_DIM
        )
        destination_is_first = not current_is_first

        entry = lane
        while entry < SPATIAL_DIM * segment_dofs:
            row = entry // segment_dofs
            column = entry - row * segment_dofs
            value = wp.float64(0.0)
            k = int(0)
            while k < SPATIAL_DIM:
                value += joint_adjoint[joint_base_row + row, k] * (
                    _matrix_value(
                        jacobian_first,
                        jacobian_second,
                        current_is_first,
                        state_base_row,
                        k,
                        column,
                    )
                    + joint_tangent[joint_base_row + k, column]
                )
                k += 1
            _write_matrix_value(
                jacobian_first,
                jacobian_second,
                destination_is_first,
                state_base_row,
                row,
                column,
                value,
            )
            entry += lane_stride

        state_row = lane
        while state_row < SPATIAL_DIM:
            source_jacobian_velocity = Vec6d()
            k = int(0)
            while k < SPATIAL_DIM:
                source_column = int(0)
                while source_column < segment_dofs:
                    source_jacobian_velocity[k] += _matrix_value(
                        jacobian_first,
                        jacobian_second,
                        current_is_first,
                        state_base_row,
                        k,
                        source_column,
                    ) * qd[environment, source_column]
                    source_column += 1
                k += 1
            derivative_value = wp.float64(0.0)
            velocity_value = wp.float64(0.0)
            gravity_value = wp.float64(0.0)
            k = int(0)
            while k < SPATIAL_DIM:
                derivative_value += (
                    joint_adjoint[joint_base_row + state_row, k]
                    * (
                        _vector_value(
                            jacobian_dot_qd_first,
                            jacobian_dot_qd_second,
                            current_is_first,
                            state_base_row,
                            k,
                        )
                        + joint_tangent_dot_qd[joint_base_row + k, 0]
                    )
                    + joint_adjoint_dot[joint_base_row + state_row, k]
                    * source_jacobian_velocity[k]
                )
                velocity_value += joint_adjoint[
                    joint_base_row + state_row, k
                ] * (
                    _vector_value(
                        velocity_first,
                        velocity_second,
                        current_is_first,
                        state_base_row,
                        k,
                    )
                    + joint_velocity[joint_base_row + k, 0]
                )
                gravity_value += joint_adjoint[
                    joint_base_row + state_row, k
                ] * _vector_value(
                    gravity_first,
                    gravity_second,
                    current_is_first,
                    state_base_row,
                    k,
                )
                k += 1
            _write_vector_value(
                jacobian_dot_qd_first,
                jacobian_dot_qd_second,
                destination_is_first,
                state_base_row,
                state_row,
                derivative_value,
            )
            _write_vector_value(
                velocity_first,
                velocity_second,
                destination_is_first,
                state_base_row,
                state_row,
                velocity_value,
            )
            _write_vector_value(
                gravity_first,
                gravity_second,
                destination_is_first,
                state_base_row,
                state_row,
                gravity_value,
            )
            state_row += lane_stride
        wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
        current_is_first = destination_is_first

        cell = int(0)
        while cell < num_cells:
            destination_is_first = not current_is_first
            cell_base_row = (
                (
                    (environment * num_segments + segment) * num_cells
                    + cell
                )
                * SPATIAL_DIM
            )

            entry = lane
            while entry < SPATIAL_DIM * segment_dofs:
                row = entry // segment_dofs
                column = entry - row * segment_dofs
                local_column = global_to_local[segment, column]
                value = wp.float64(0.0)
                k = int(0)
                while k < SPATIAL_DIM:
                    tangent_value = wp.float64(0.0)
                    if local_column >= 0:
                        tangent_value = cell_tangent_local[
                            cell_base_row + k, local_column
                        ]
                    value += cell_adjoint[cell_base_row + row, k] * (
                        _matrix_value(
                            jacobian_first,
                            jacobian_second,
                            current_is_first,
                            state_base_row,
                            k,
                            column,
                        )
                        + tangent_value
                    )
                    k += 1
                _write_matrix_value(
                    jacobian_first,
                    jacobian_second,
                    destination_is_first,
                    state_base_row,
                    row,
                    column,
                    value,
                )
                entry += lane_stride

            state_row = lane
            while state_row < SPATIAL_DIM:
                source_jacobian_velocity = Vec6d()
                k = int(0)
                while k < SPATIAL_DIM:
                    source_column = int(0)
                    while source_column < segment_dofs:
                        source_jacobian_velocity[k] += _matrix_value(
                            jacobian_first,
                            jacobian_second,
                            current_is_first,
                            state_base_row,
                            k,
                            source_column,
                        ) * qd[environment, source_column]
                        source_column += 1
                    k += 1
                transported_source_velocity = Vec6d()
                target = int(0)
                while target < SPATIAL_DIM:
                    k = int(0)
                    while k < SPATIAL_DIM:
                        transported_source_velocity[target] += (
                            cell_adjoint[cell_base_row + target, k]
                            * source_jacobian_velocity[k]
                        )
                        k += 1
                    target += 1
                derivative_value = wp.float64(0.0)
                velocity_value = wp.float64(0.0)
                gravity_value = wp.float64(0.0)
                k = int(0)
                while k < SPATIAL_DIM:
                    adjoint_value = cell_adjoint[
                        cell_base_row + state_row, k
                    ]
                    derivative_value += adjoint_value * (
                        _vector_value(
                            jacobian_dot_qd_first,
                            jacobian_dot_qd_second,
                            current_is_first,
                            state_base_row,
                            k,
                        )
                        + cell_tangent_velocity_dot[cell_base_row + k, 0]
                    )
                    velocity_value += adjoint_value * (
                        _vector_value(
                            velocity_first,
                            velocity_second,
                            current_is_first,
                            state_base_row,
                            k,
                        )
                        + cell_link_velocity[cell_base_row + k, 0]
                    )
                    gravity_value += adjoint_value * _vector_value(
                        gravity_first,
                        gravity_second,
                        current_is_first,
                        state_base_row,
                        k,
                    )
                    k += 1
                step = Vec6d(
                    cell_step_velocity[cell_base_row + 0, 0],
                    cell_step_velocity[cell_base_row + 1, 0],
                    cell_step_velocity[cell_base_row + 2, 0],
                    cell_step_velocity[cell_base_row + 3, 0],
                    cell_step_velocity[cell_base_row + 4, 0],
                    cell_step_velocity[cell_base_row + 5, 0],
                )
                bracket = _ad_action(step, transported_source_velocity)
                _write_vector_value(
                    jacobian_dot_qd_first,
                    jacobian_dot_qd_second,
                    destination_is_first,
                    state_base_row,
                    state_row,
                    derivative_value - bracket[state_row],
                )
                _write_vector_value(
                    velocity_first,
                    velocity_second,
                    destination_is_first,
                    state_base_row,
                    state_row,
                    velocity_value,
                )
                _write_vector_value(
                    gravity_first,
                    gravity_second,
                    destination_is_first,
                    state_base_row,
                    state_row,
                    gravity_value,
                )
                state_row += lane_stride
            wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
            current_is_first = destination_is_first

            if cell < num_quadrature:
                quadrature_item = segment * num_quadrature + cell
                weight = weights[quadrature_item]
                entry = lane
                while entry < segment_dofs * segment_dofs:
                    output_row = entry // segment_dofs
                    output_column = entry - output_row * segment_dofs
                    value = wp.float64(0.0)
                    row = int(0)
                    while row < SPATIAL_DIM:
                        value += (
                            _matrix_value(
                                jacobian_first,
                                jacobian_second,
                                current_is_first,
                                state_base_row,
                                row,
                                output_row,
                            )
                            * weight
                            * masses[quadrature_item, row]
                            * _matrix_value(
                                jacobian_first,
                                jacobian_second,
                                current_is_first,
                                state_base_row,
                                row,
                                output_column,
                            )
                        )
                        row += 1
                    inertia[environment, output_row, output_column] += value
                    entry += lane_stride

                column = lane
                while column < segment_dofs:
                    wrench = _coadjoint_wrench(
                        velocity_first,
                        velocity_second,
                        jacobian_dot_qd_first,
                        jacobian_dot_qd_second,
                        current_is_first,
                        state_base_row,
                        masses,
                        quadrature_item,
                    )
                    coriolis_value = wp.float64(0.0)
                    gravity_value = wp.float64(0.0)
                    row = int(0)
                    while row < SPATIAL_DIM:
                        jacobian_value = _matrix_value(
                            jacobian_first,
                            jacobian_second,
                            current_is_first,
                            state_base_row,
                            row,
                            column,
                        )
                        coriolis_value += (
                            weight * jacobian_value * wrench[row]
                        )
                        gravity_value -= (
                            weight
                            * jacobian_value
                            * masses[quadrature_item, row]
                            * _vector_value(
                                gravity_first,
                                gravity_second,
                                current_is_first,
                                state_base_row,
                                row,
                            )
                        )
                        row += 1
                    coriolis_qd[environment, column] += coriolis_value
                    gravity_force[environment, column] += gravity_value
                    column += lane_stride
            wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
            cell += 1
        segment += 1


def _scalable_persistent_chain(
    joint_adjoint: wp.array2d(dtype=wp.float64),
    joint_adjoint_dot: wp.array2d(dtype=wp.float64),
    joint_tangent: wp.array2d(dtype=wp.float64),
    joint_tangent_dot_qd: wp.array2d(dtype=wp.float64),
    joint_velocity: wp.array2d(dtype=wp.float64),
    cell_adjoint: wp.array2d(dtype=wp.float64),
    cell_tangent_local: wp.array2d(dtype=wp.float64),
    cell_link_velocity: wp.array2d(dtype=wp.float64),
    cell_step_velocity: wp.array2d(dtype=wp.float64),
    cell_tangent_velocity_dot: wp.array2d(dtype=wp.float64),
    global_to_local: wp.array2d(dtype=wp.int32),
    active_dofs: wp.array(dtype=wp.int32),
    qd: wp.array2d(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    masses: wp.array2d(dtype=wp.float64),
    gravity_base: wp.array(dtype=wp.float64),
    num_cells: wp.array(dtype=wp.int32),
    num_quadrature: wp.array(dtype=wp.int32),
    lanes_per_block: wp.array(dtype=wp.int32),
    jacobian_first: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_first: wp.array2d(dtype=wp.float64),
    velocity_first: wp.array2d(dtype=wp.float64),
    gravity_first: wp.array2d(dtype=wp.float64),
    jacobian_second: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_second: wp.array2d(dtype=wp.float64),
    velocity_second: wp.array2d(dtype=wp.float64),
    gravity_second: wp.array2d(dtype=wp.float64),
    inertia: wp.array3d(dtype=wp.float64),
    coriolis_qd: wp.array2d(dtype=wp.float64),
    gravity_force: wp.array2d(dtype=wp.float64),
):
    wp.launch_tiled(
        scalable_persistent_chain_kernel,
        dim=qd.shape[0],
        inputs=[
            joint_adjoint,
            joint_adjoint_dot,
            joint_tangent,
            joint_tangent_dot_qd,
            joint_velocity,
            cell_adjoint,
            cell_tangent_local,
            cell_link_velocity,
            cell_step_velocity,
            cell_tangent_velocity_dot,
            global_to_local,
            active_dofs,
            qd,
            weights,
            masses,
            gravity_base,
            num_cells,
            num_quadrature,
            lanes_per_block,
        ],
        outputs=[
            jacobian_first,
            jacobian_dot_qd_first,
            velocity_first,
            gravity_first,
            jacobian_second,
            jacobian_dot_qd_second,
            velocity_second,
            gravity_second,
            inertia,
            coriolis_qd,
            gravity_force,
        ],
        block_dim=BLOCK_DIM,
    )


scalable_persistent_chain = wp.jax_callable(
    _scalable_persistent_chain, num_outputs=11
)
