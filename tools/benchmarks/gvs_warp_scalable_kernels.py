# ruff: noqa: I001, UP018
"""Shape-generic Warp kernels for scalable GVS compilation experiments.

The kernels keep spatial dimension six, but obtain the number of generalized
coordinates and local strain coordinates from array shapes. Runtime ``while``
loops prevent Warp from expanding work proportional to the number of segments
into the generated CUDA source.
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})

from tools.benchmarks.gvs_warp_lie_kernels import (
    Vec6d,
    _ad_action,
    _adjoint_inverse_action,
    _adjoint_inverse_entry,
    _forward_coefficients,
    _left_action,
    _left_coefficient_x_derivatives,
    _left_coefficients,
    _left_total_derivative_action,
    _translation,
)


SPATIAL_DIM = 6


@wp.func
def _load_dynamic_column(
    matrix: wp.array2d(dtype=wp.float64), base_row: int, column: int
) -> Vec6d:
    return Vec6d(
        matrix[base_row + 0, column],
        matrix[base_row + 1, column],
        matrix[base_row + 2, column],
        matrix[base_row + 3, column],
        matrix[base_row + 4, column],
        matrix[base_row + 5, column],
    )


@wp.func
def _dynamic_strain(
    basis: wp.array2d(dtype=wp.float64),
    reference: wp.array2d(dtype=wp.float64),
    coordinates: wp.array2d(dtype=wp.float64),
    cell_item: int,
    coordinate_item: int,
) -> Vec6d:
    base_row = cell_item * SPATIAL_DIM
    value = Vec6d(
        reference[cell_item, 0],
        reference[cell_item, 1],
        reference[cell_item, 2],
        reference[cell_item, 3],
        reference[cell_item, 4],
        reference[cell_item, 5],
    )
    column = int(0)
    while column < coordinates.shape[1]:
        coordinate = coordinates[coordinate_item, column]
        row = int(0)
        while row < SPATIAL_DIM:
            value[row] += basis[base_row + row, column] * coordinate
            row += 1
        column += 1
    return value


@wp.func
def _dynamic_strain_rate(
    basis: wp.array2d(dtype=wp.float64),
    velocities: wp.array2d(dtype=wp.float64),
    cell_item: int,
    coordinate_item: int,
) -> Vec6d:
    base_row = cell_item * SPATIAL_DIM
    value = Vec6d()
    column = int(0)
    while column < velocities.shape[1]:
        velocity = velocities[coordinate_item, column]
        row = int(0)
        while row < SPATIAL_DIM:
            value[row] += basis[base_row + row, column] * velocity
            row += 1
        column += 1
    return value


@wp.kernel(enable_backward=False)
def scalable_cell_terms_kernel(
    q_link: wp.array2d(dtype=wp.float64),
    qd_link: wp.array2d(dtype=wp.float64),
    basis_z1: wp.array2d(dtype=wp.float64),
    basis_z2: wp.array2d(dtype=wp.float64),
    reference_z1: wp.array2d(dtype=wp.float64),
    reference_z2: wp.array2d(dtype=wp.float64),
    segment_lengths: wp.array(dtype=wp.float64),
    cell_widths: wp.array(dtype=wp.float64),
    num_cells: wp.array(dtype=wp.int32),
    order_zero: wp.array(dtype=wp.int32),
    adjoint: wp.array2d(dtype=wp.float64),
    tangent_local: wp.array2d(dtype=wp.float64),
    link_velocity: wp.array2d(dtype=wp.float64),
    step_velocity: wp.array2d(dtype=wp.float64),
    tangent_velocity_dot: wp.array2d(dtype=wp.float64),
):
    """Compute general cell-local Lie data with an optional order-zero branch."""

    work_item = wp.tid()
    cells_per_segment = num_cells[0]
    cells_per_environment = segment_lengths.shape[0] * cells_per_segment
    environment = work_item // cells_per_environment
    environment_cell = work_item - environment * cells_per_environment
    segment = environment_cell // cells_per_segment
    cell = environment_cell - segment * cells_per_segment
    cell_item = segment * cells_per_segment + cell
    coordinate_item = environment * segment_lengths.shape[0] + segment
    base_row = cell_item * SPATIAL_DIM
    output_base_row = work_item * SPATIAL_DIM

    xi1 = _dynamic_strain(
        basis_z1, reference_z1, q_link, cell_item, coordinate_item
    )
    xid1 = _dynamic_strain_rate(
        basis_z1, qd_link, cell_item, coordinate_item
    )
    length = segment_lengths[segment]
    width = cell_widths[cell_item]
    alpha = length * width * wp.float64(0.5)
    commutator_coefficient = (
        wp.sqrt(wp.float64(3.0))
        * length
        * length
        * width
        * width
        / wp.float64(12.0)
    )

    xi2 = xi1
    xid2 = xid1
    magnus = length * width * xi1
    magnus_dot = length * width * xid1
    if order_zero[0] == 0:
        xi2 = _dynamic_strain(
            basis_z2, reference_z2, q_link, cell_item, coordinate_item
        )
        xid2 = _dynamic_strain_rate(
            basis_z2, qd_link, cell_item, coordinate_item
        )
        magnus = alpha * (xi1 + xi2) + commutator_coefficient * _ad_action(
            xi1, xi2
        )
        magnus_dot = alpha * (xid1 + xid2) + commutator_coefficient * (
            _ad_action(xid1, xi2) + _ad_action(xi1, xid2)
        )

    angle_sq = (
        magnus[0] * magnus[0]
        + magnus[1] * magnus[1]
        + magnus[2] * magnus[2]
    )
    coefficients = _left_coefficients(angle_sq)
    omega = wp.vec3d(magnus[0], magnus[1], magnus[2])
    linear = wp.vec3d(magnus[3], magnus[4], magnus[5])
    forward = _forward_coefficients(angle_sq)
    translation = _translation(omega, linear, forward)

    row = int(0)
    while row < SPATIAL_DIM:
        column = int(0)
        while column < SPATIAL_DIM:
            adjoint[output_base_row + row, column] = _adjoint_inverse_entry(
                omega,
                translation,
                angle_sq,
                forward,
                row,
                column,
            )
            column += 1
        row += 1

    local_column = int(0)
    while local_column < q_link.shape[1]:
        basis1 = _load_dynamic_column(basis_z1, base_row, local_column)
        magnus_basis = length * width * basis1
        if order_zero[0] == 0:
            basis2 = _load_dynamic_column(basis_z2, base_row, local_column)
            magnus_basis = alpha * (
                basis1 + basis2
            ) + commutator_coefficient * (
                _ad_action(xi1, basis2) - _ad_action(xi2, basis1)
            )
        local_tangent = _left_action(magnus, magnus_basis, coefficients)
        row = int(0)
        while row < SPATIAL_DIM:
            tangent_local[output_base_row + row, local_column] = local_tangent[row]
            row += 1
        local_column += 1

    link = _left_action(magnus, magnus_dot, coefficients)
    step = _adjoint_inverse_action(
        omega, translation, angle_sq, forward, link
    )
    magnus_basis_dot_qd = Vec6d()
    if order_zero[0] == 0:
        magnus_basis_dot_qd = (
            wp.float64(2.0)
            * commutator_coefficient
            * _ad_action(xid1, xid2)
        )
    coefficient_derivatives = _left_coefficient_x_derivatives(angle_sq)
    angle_sq_dot = wp.float64(2.0) * (
        magnus[0] * magnus_dot[0]
        + magnus[1] * magnus_dot[1]
        + magnus[2] * magnus_dot[2]
    )
    tangent_dot_velocity_value = _left_total_derivative_action(
        magnus,
        magnus_dot,
        magnus_dot,
        magnus_basis_dot_qd,
        coefficients,
        coefficient_derivatives * angle_sq_dot,
    )
    row = int(0)
    while row < SPATIAL_DIM:
        link_velocity[output_base_row + row, 0] = link[row]
        step_velocity[output_base_row + row, 0] = step[row]
        tangent_velocity_dot[output_base_row + row, 0] = (
            tangent_dot_velocity_value[row]
        )
        row += 1


def _scalable_cell_terms(
    q_link: wp.array2d(dtype=wp.float64),
    qd_link: wp.array2d(dtype=wp.float64),
    basis_z1: wp.array2d(dtype=wp.float64),
    basis_z2: wp.array2d(dtype=wp.float64),
    reference_z1: wp.array2d(dtype=wp.float64),
    reference_z2: wp.array2d(dtype=wp.float64),
    segment_lengths: wp.array(dtype=wp.float64),
    cell_widths: wp.array(dtype=wp.float64),
    num_cells: wp.array(dtype=wp.int32),
    order_zero: wp.array(dtype=wp.int32),
    adjoint: wp.array2d(dtype=wp.float64),
    tangent_local: wp.array2d(dtype=wp.float64),
    link_velocity: wp.array2d(dtype=wp.float64),
    step_velocity: wp.array2d(dtype=wp.float64),
    tangent_velocity_dot: wp.array2d(dtype=wp.float64),
):
    # ``num_cells`` has one entry whose value is not available to Python here.
    # Derive the work count from the reference-strain rows instead.
    work_items = (q_link.shape[0] * reference_z1.shape[0]) // segment_lengths.shape[0]
    wp.launch(
        scalable_cell_terms_kernel,
        dim=work_items,
        inputs=[
            q_link,
            qd_link,
            basis_z1,
            basis_z2,
            reference_z1,
            reference_z2,
            segment_lengths,
            cell_widths,
            num_cells,
            order_zero,
        ],
        outputs=[
            adjoint,
            tangent_local,
            link_velocity,
            step_velocity,
            tangent_velocity_dot,
        ],
        block_dim=128,
    )


scalable_cell_terms = wp.jax_callable(_scalable_cell_terms, num_outputs=5)


@wp.func
def _source_matrix_value(
    initial: wp.array2d(dtype=wp.float64),
    states: wp.array2d(dtype=wp.float64),
    environment: int,
    cell: int,
    num_cells: int,
    row: int,
    column: int,
) -> wp.float64:
    if cell == 0:
        return initial[environment * SPATIAL_DIM + row, column]
    state_row = (
        (environment * num_cells + cell - 1) * SPATIAL_DIM + row
    )
    return states[state_row, column]


@wp.func
def _source_vector_value(
    initial: wp.array2d(dtype=wp.float64),
    states: wp.array2d(dtype=wp.float64),
    environment: int,
    cell: int,
    num_cells: int,
    row: int,
) -> wp.float64:
    if cell == 0:
        return initial[environment * SPATIAL_DIM + row, 0]
    state_row = (
        (environment * num_cells + cell - 1) * SPATIAL_DIM + row
    )
    return states[state_row, 0]


@wp.kernel(enable_backward=False)
def scalable_segment_cell_kernel(
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
    cell: int,
    jacobian_states: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_states: wp.array2d(dtype=wp.float64),
    velocity_states: wp.array2d(dtype=wp.float64),
    gravity_states: wp.array2d(dtype=wp.float64),
):
    """Advance one cell with runtime global and local coordinate counts."""

    linear_index = wp.tid()
    num_dofs = qd.shape[1]
    num_cells = adjoint.shape[0] // (qd.shape[0] * SPATIAL_DIM)
    entries_per_environment = SPATIAL_DIM * num_dofs
    environment = linear_index // entries_per_environment
    entry = linear_index - environment * entries_per_environment
    row = entry // num_dofs
    column = entry - row * num_dofs
    work_item = environment * num_cells + cell
    cell_base_row = work_item * SPATIAL_DIM

    jacobian_value = wp.float64(0.0)
    k = int(0)
    while k < SPATIAL_DIM:
        tangent_value = wp.float64(0.0)
        local_column = global_to_local[column]
        if local_column >= 0:
            tangent_value = tangent_local[cell_base_row + k, local_column]
        jacobian_value += adjoint[cell_base_row + row, k] * (
            _source_matrix_value(
                jacobian_initial,
                jacobian_states,
                environment,
                cell,
                num_cells,
                k,
                column,
            )
            + tangent_value
        )
        k += 1
    state_row = cell_base_row + row
    jacobian_states[state_row, column] = jacobian_value

    if column == 0:
        source_velocity = Vec6d()
        k = int(0)
        while k < SPATIAL_DIM:
            source_column = int(0)
            while source_column < num_dofs:
                source_velocity[k] += _source_matrix_value(
                    jacobian_initial,
                    jacobian_states,
                    environment,
                    cell,
                    num_cells,
                    k,
                    source_column,
                ) * qd[environment, source_column]
                source_column += 1
            k += 1

        transported_velocity = Vec6d()
        derivative_value = wp.float64(0.0)
        velocity_value = wp.float64(0.0)
        gravity_value = wp.float64(0.0)
        k = int(0)
        while k < SPATIAL_DIM:
            adjoint_value = adjoint[cell_base_row + row, k]
            derivative_value += adjoint_value * (
                _source_vector_value(
                    jacobian_dot_qd_initial,
                    jacobian_dot_qd_states,
                    environment,
                    cell,
                    num_cells,
                    k,
                )
                + tangent_velocity_dot[cell_base_row + k, 0]
            )
            velocity_value += adjoint_value * (
                _source_vector_value(
                    velocity_initial,
                    velocity_states,
                    environment,
                    cell,
                    num_cells,
                    k,
                )
                + link_velocity[cell_base_row + k, 0]
            )
            gravity_value += adjoint_value * _source_vector_value(
                gravity_initial,
                gravity_states,
                environment,
                cell,
                num_cells,
                k,
            )

            target = int(0)
            while target < SPATIAL_DIM:
                transported_velocity[target] += (
                    adjoint[cell_base_row + target, k] * source_velocity[k]
                )
                target += 1
            k += 1

        step = Vec6d(
            step_velocity[cell_base_row + 0, 0],
            step_velocity[cell_base_row + 1, 0],
            step_velocity[cell_base_row + 2, 0],
            step_velocity[cell_base_row + 3, 0],
            step_velocity[cell_base_row + 4, 0],
            step_velocity[cell_base_row + 5, 0],
        )
        bracket = _ad_action(step, transported_velocity)
        jacobian_dot_qd_states[state_row, 0] = derivative_value - bracket[row]
        velocity_states[state_row, 0] = velocity_value
        gravity_states[state_row, 0] = gravity_value


def _scalable_segment_recurrence(
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
    jacobian_states: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_states: wp.array2d(dtype=wp.float64),
    velocity_states: wp.array2d(dtype=wp.float64),
    gravity_states: wp.array2d(dtype=wp.float64),
):
    batch_size = qd.shape[0]
    num_dofs = qd.shape[1]
    num_cells = adjoint.shape[0] // (batch_size * SPATIAL_DIM)
    for cell in range(num_cells):
        wp.launch(
            scalable_segment_cell_kernel,
            dim=batch_size * SPATIAL_DIM * num_dofs,
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
                cell,
            ],
            outputs=[
                jacobian_states,
                jacobian_dot_qd_states,
                velocity_states,
                gravity_states,
            ],
            block_dim=128,
        )


scalable_segment_recurrence = wp.jax_callable(
    _scalable_segment_recurrence, num_outputs=4
)
