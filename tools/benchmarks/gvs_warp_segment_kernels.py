# ruff: noqa: I001, UP018
"""Runtime-sized cooperative GVS segment recurrence and assembly."""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})

from tools.benchmarks.gvs_warp_lie_kernels import Vec6d, _ad_action


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


@wp.kernel(enable_backward=False)
def scalable_cooperative_segment_kernel(
    adjoint: wp.array2d(dtype=wp.float64),
    tangent_local: wp.array2d(dtype=wp.float64),
    link_velocity: wp.array2d(dtype=wp.float64),
    step_velocity: wp.array2d(dtype=wp.float64),
    tangent_velocity_dot: wp.array2d(dtype=wp.float64),
    global_to_local: wp.array(dtype=wp.int32),
    qd: wp.array2d(dtype=wp.float64),
    jacobian_initial: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_initial: wp.array2d(dtype=wp.float64),
    velocity_initial: wp.array2d(dtype=wp.float64),
    gravity_initial: wp.array2d(dtype=wp.float64),
    joint_adjoint: wp.array2d(dtype=wp.float64),
    joint_adjoint_dot: wp.array2d(dtype=wp.float64),
    joint_tangent: wp.array2d(dtype=wp.float64),
    joint_tangent_dot_qd: wp.array2d(dtype=wp.float64),
    joint_velocity: wp.array2d(dtype=wp.float64),
    apply_joint: wp.array(dtype=wp.int32),
    weights: wp.array(dtype=wp.float64),
    masses: wp.array2d(dtype=wp.float64),
    lanes_per_block: wp.array(dtype=wp.int32),
    jacobian_tip: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_tip: wp.array2d(dtype=wp.float64),
    velocity_tip: wp.array2d(dtype=wp.float64),
    gravity_tip: wp.array2d(dtype=wp.float64),
    inertia: wp.array3d(dtype=wp.float64),
    coriolis_qd: wp.array2d(dtype=wp.float64),
    gravity_force: wp.array2d(dtype=wp.float64),
    jacobian_scratch: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_scratch: wp.array2d(dtype=wp.float64),
    velocity_scratch: wp.array2d(dtype=wp.float64),
    gravity_scratch: wp.array2d(dtype=wp.float64),
):
    """Advance one environment/segment per block and assemble online."""

    environment, lane = wp.tid()
    num_dofs = qd.shape[1]
    num_cells = adjoint.shape[0] // (qd.shape[0] * SPATIAL_DIM)
    lane_stride = lanes_per_block[0]
    state_base_row = environment * SPATIAL_DIM
    barrier = wp.tile_zeros(shape=(1,), dtype=wp.int32, storage="shared")

    entry = lane
    while entry < SPATIAL_DIM * num_dofs:
        row = entry // num_dofs
        column = entry - row * num_dofs
        value = jacobian_initial[state_base_row + row, column]
        if apply_joint[0] != 0:
            value = wp.float64(0.0)
            k = int(0)
            while k < SPATIAL_DIM:
                value += joint_adjoint[state_base_row + row, k] * (
                    jacobian_initial[state_base_row + k, column]
                    + joint_tangent[state_base_row + k, column]
                )
                k += 1
        jacobian_tip[state_base_row + row, column] = value
        entry += lane_stride
    row = lane
    while row < SPATIAL_DIM:
        derivative_value = jacobian_dot_qd_initial[
            state_base_row + row, 0
        ]
        velocity_value = velocity_initial[state_base_row + row, 0]
        gravity_value = gravity_initial[state_base_row + row, 0]
        if apply_joint[0] != 0:
            source_jacobian_velocity = Vec6d()
            k = int(0)
            while k < SPATIAL_DIM:
                column = int(0)
                while column < num_dofs:
                    source_jacobian_velocity[k] += jacobian_initial[
                        state_base_row + k, column
                    ] * qd[environment, column]
                    column += 1
                k += 1
            derivative_value = wp.float64(0.0)
            velocity_value = wp.float64(0.0)
            gravity_value = wp.float64(0.0)
            k = int(0)
            while k < SPATIAL_DIM:
                derivative_value += (
                    joint_adjoint[state_base_row + row, k]
                    * (
                        jacobian_dot_qd_initial[state_base_row + k, 0]
                        + joint_tangent_dot_qd[state_base_row + k, 0]
                    )
                    + joint_adjoint_dot[state_base_row + row, k]
                    * source_jacobian_velocity[k]
                )
                velocity_value += joint_adjoint[
                    state_base_row + row, k
                ] * (
                    velocity_initial[state_base_row + k, 0]
                    + joint_velocity[state_base_row + k, 0]
                )
                gravity_value += joint_adjoint[
                    state_base_row + row, k
                ] * gravity_initial[state_base_row + k, 0]
                k += 1
        jacobian_dot_qd_tip[state_base_row + row, 0] = derivative_value
        velocity_tip[state_base_row + row, 0] = velocity_value
        gravity_tip[state_base_row + row, 0] = gravity_value
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

    cell = int(0)
    while cell < num_cells:
        source_is_tip = cell % 2 == 0
        destination_is_tip = not source_is_tip
        cell_base_row = (environment * num_cells + cell) * SPATIAL_DIM

        entry = lane
        while entry < SPATIAL_DIM * num_dofs:
            row = entry // num_dofs
            column = entry - row * num_dofs
            local_column = global_to_local[column]
            value = wp.float64(0.0)
            k = int(0)
            while k < SPATIAL_DIM:
                tangent_value = wp.float64(0.0)
                if local_column >= 0:
                    tangent_value = tangent_local[
                        cell_base_row + k, local_column
                    ]
                value += adjoint[cell_base_row + row, k] * (
                    _matrix_value(
                        jacobian_tip,
                        jacobian_scratch,
                        source_is_tip,
                        state_base_row,
                        k,
                        column,
                    )
                    + tangent_value
                )
                k += 1
            _write_matrix_value(
                jacobian_tip,
                jacobian_scratch,
                destination_is_tip,
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
                while source_column < num_dofs:
                    source_jacobian_velocity[k] += _matrix_value(
                        jacobian_tip,
                        jacobian_scratch,
                        source_is_tip,
                        state_base_row,
                        k,
                        source_column,
                    ) * qd[environment, source_column]
                    source_column += 1
                k += 1

            transported_source_velocity = Vec6d()
            derivative_value = wp.float64(0.0)
            velocity_value = wp.float64(0.0)
            gravity_value = wp.float64(0.0)
            k = int(0)
            while k < SPATIAL_DIM:
                adjoint_value = adjoint[cell_base_row + state_row, k]
                derivative_value += adjoint_value * (
                    _vector_value(
                        jacobian_dot_qd_tip,
                        jacobian_dot_qd_scratch,
                        source_is_tip,
                        state_base_row,
                        k,
                    )
                    + tangent_velocity_dot[cell_base_row + k, 0]
                )
                velocity_value += adjoint_value * (
                    _vector_value(
                        velocity_tip,
                        velocity_scratch,
                        source_is_tip,
                        state_base_row,
                        k,
                    )
                    + link_velocity[cell_base_row + k, 0]
                )
                gravity_value += adjoint_value * _vector_value(
                    gravity_tip,
                    gravity_scratch,
                    source_is_tip,
                    state_base_row,
                    k,
                )
                k += 1
            target = int(0)
            while target < SPATIAL_DIM:
                k = int(0)
                while k < SPATIAL_DIM:
                    transported_source_velocity[target] += (
                        adjoint[cell_base_row + target, k]
                        * source_jacobian_velocity[k]
                    )
                    k += 1
                target += 1
            step = Vec6d(
                step_velocity[cell_base_row + 0, 0],
                step_velocity[cell_base_row + 1, 0],
                step_velocity[cell_base_row + 2, 0],
                step_velocity[cell_base_row + 3, 0],
                step_velocity[cell_base_row + 4, 0],
                step_velocity[cell_base_row + 5, 0],
            )
            bracket = _ad_action(step, transported_source_velocity)
            _write_vector_value(
                jacobian_dot_qd_tip,
                jacobian_dot_qd_scratch,
                destination_is_tip,
                state_base_row,
                state_row,
                derivative_value - bracket[state_row],
            )
            _write_vector_value(
                velocity_tip,
                velocity_scratch,
                destination_is_tip,
                state_base_row,
                state_row,
                velocity_value,
            )
            _write_vector_value(
                gravity_tip,
                gravity_scratch,
                destination_is_tip,
                state_base_row,
                state_row,
                gravity_value,
            )
            state_row += lane_stride
        wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)

        if cell < weights.shape[0]:
            weight = weights[cell]
            entry = lane
            while entry < num_dofs * num_dofs:
                output_row = entry // num_dofs
                output_column = entry - output_row * num_dofs
                value = wp.float64(0.0)
                row = int(0)
                while row < SPATIAL_DIM:
                    value += (
                        _matrix_value(
                            jacobian_tip,
                            jacobian_scratch,
                            destination_is_tip,
                            state_base_row,
                            row,
                            output_row,
                        )
                        * weight
                        * masses[cell, row]
                        * _matrix_value(
                            jacobian_tip,
                            jacobian_scratch,
                            destination_is_tip,
                            state_base_row,
                            row,
                            output_column,
                        )
                    )
                    row += 1
                inertia[environment, output_row, output_column] += value
                entry += lane_stride

            column = lane
            while column < num_dofs:
                wrench = _coadjoint_wrench(
                    velocity_tip,
                    velocity_scratch,
                    jacobian_dot_qd_tip,
                    jacobian_dot_qd_scratch,
                    destination_is_tip,
                    state_base_row,
                    masses,
                    cell,
                )
                coriolis_value = wp.float64(0.0)
                gravity_value = wp.float64(0.0)
                row = int(0)
                while row < SPATIAL_DIM:
                    jacobian_value = _matrix_value(
                        jacobian_tip,
                        jacobian_scratch,
                        destination_is_tip,
                        state_base_row,
                        row,
                        column,
                    )
                    coriolis_value += weight * jacobian_value * wrench[row]
                    gravity_value -= (
                        weight
                        * jacobian_value
                        * masses[cell, row]
                        * _vector_value(
                            gravity_tip,
                            gravity_scratch,
                            destination_is_tip,
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

    if num_cells % 2 == 1:
        entry = lane
        while entry < SPATIAL_DIM * num_dofs:
            row = entry // num_dofs
            column = entry - row * num_dofs
            jacobian_tip[state_base_row + row, column] = jacobian_scratch[
                state_base_row + row, column
            ]
            entry += lane_stride
        row = lane
        while row < SPATIAL_DIM:
            jacobian_dot_qd_tip[state_base_row + row, 0] = (
                jacobian_dot_qd_scratch[state_base_row + row, 0]
            )
            velocity_tip[state_base_row + row, 0] = velocity_scratch[
                state_base_row + row, 0
            ]
            gravity_tip[state_base_row + row, 0] = gravity_scratch[
                state_base_row + row, 0
            ]
            row += lane_stride


def _scalable_cooperative_segment(
    adjoint: wp.array2d(dtype=wp.float64),
    tangent_local: wp.array2d(dtype=wp.float64),
    link_velocity: wp.array2d(dtype=wp.float64),
    step_velocity: wp.array2d(dtype=wp.float64),
    tangent_velocity_dot: wp.array2d(dtype=wp.float64),
    global_to_local: wp.array(dtype=wp.int32),
    qd: wp.array2d(dtype=wp.float64),
    jacobian_initial: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_initial: wp.array2d(dtype=wp.float64),
    velocity_initial: wp.array2d(dtype=wp.float64),
    gravity_initial: wp.array2d(dtype=wp.float64),
    joint_adjoint: wp.array2d(dtype=wp.float64),
    joint_adjoint_dot: wp.array2d(dtype=wp.float64),
    joint_tangent: wp.array2d(dtype=wp.float64),
    joint_tangent_dot_qd: wp.array2d(dtype=wp.float64),
    joint_velocity: wp.array2d(dtype=wp.float64),
    apply_joint: wp.array(dtype=wp.int32),
    weights: wp.array(dtype=wp.float64),
    masses: wp.array2d(dtype=wp.float64),
    lanes_per_block: wp.array(dtype=wp.int32),
    jacobian_tip: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_tip: wp.array2d(dtype=wp.float64),
    velocity_tip: wp.array2d(dtype=wp.float64),
    gravity_tip: wp.array2d(dtype=wp.float64),
    inertia: wp.array3d(dtype=wp.float64),
    coriolis_qd: wp.array2d(dtype=wp.float64),
    gravity_force: wp.array2d(dtype=wp.float64),
    jacobian_scratch: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_scratch: wp.array2d(dtype=wp.float64),
    velocity_scratch: wp.array2d(dtype=wp.float64),
    gravity_scratch: wp.array2d(dtype=wp.float64),
):
    wp.launch_tiled(
        scalable_cooperative_segment_kernel,
        dim=qd.shape[0],
        inputs=[
            adjoint,
            tangent_local,
            link_velocity,
            step_velocity,
            tangent_velocity_dot,
            global_to_local,
            qd,
            jacobian_initial,
            jacobian_dot_qd_initial,
            velocity_initial,
            gravity_initial,
            joint_adjoint,
            joint_adjoint_dot,
            joint_tangent,
            joint_tangent_dot_qd,
            joint_velocity,
            apply_joint,
            weights,
            masses,
            lanes_per_block,
        ],
        outputs=[
            jacobian_tip,
            jacobian_dot_qd_tip,
            velocity_tip,
            gravity_tip,
            inertia,
            coriolis_qd,
            gravity_force,
            jacobian_scratch,
            jacobian_dot_qd_scratch,
            velocity_scratch,
            gravity_scratch,
        ],
        block_dim=BLOCK_DIM,
    )


scalable_cooperative_segment = wp.jax_callable(
    _scalable_cooperative_segment, num_outputs=11
)
