# ruff: noqa: I001, UP018
"""Matrix-free local GVS Lie-algebra preparation kernels.

This fixed-shape prototype computes the cell-local quantities consumed by the
recurrence directly from link coordinates, velocities, bases, and reference
strains.  It deliberately applies ``ad`` polynomials to vectors instead of
materializing dense ``ad``, tangent, or tangent-derivative matrices.
"""

from __future__ import annotations

import warp as wp

from tools.benchmarks.gvs_warp_kernels import (
    MAX_DOF,
    NUM_CELLS,
    NUM_DOFS,
    NUM_SEGMENTS,
    SPATIAL_DIM,
)


Vec6d = wp.types.vector(length=6, dtype=wp.float64)


@wp.func
def _ad_action(left: Vec6d, right: Vec6d) -> Vec6d:
    left_omega = wp.vec3d(left[0], left[1], left[2])
    left_linear = wp.vec3d(left[3], left[4], left[5])
    right_omega = wp.vec3d(right[0], right[1], right[2])
    right_linear = wp.vec3d(right[3], right[4], right[5])
    angular = wp.cross(left_omega, right_omega)
    linear = wp.cross(left_linear, right_omega) + wp.cross(
        left_omega, right_linear
    )
    return Vec6d(
        angular[0],
        angular[1],
        angular[2],
        linear[0],
        linear[1],
        linear[2],
    )


@wp.func
def _load_column(
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
def _strain(
    basis: wp.array2d(dtype=wp.float64),
    reference: wp.array2d(dtype=wp.float64),
    coordinates: wp.array2d(dtype=wp.float64),
    cell_item: int,
    coordinate_item: int,
) -> Vec6d:
    base_row = cell_item * SPATIAL_DIM
    value0 = reference[cell_item, 0]
    value1 = reference[cell_item, 1]
    value2 = reference[cell_item, 2]
    value3 = reference[cell_item, 3]
    value4 = reference[cell_item, 4]
    value5 = reference[cell_item, 5]
    for column in range(MAX_DOF):
        coordinate = coordinates[coordinate_item, column]
        value0 += basis[base_row + 0, column] * coordinate
        value1 += basis[base_row + 1, column] * coordinate
        value2 += basis[base_row + 2, column] * coordinate
        value3 += basis[base_row + 3, column] * coordinate
        value4 += basis[base_row + 4, column] * coordinate
        value5 += basis[base_row + 5, column] * coordinate
    return Vec6d(value0, value1, value2, value3, value4, value5)


@wp.func
def _strain_rate(
    basis: wp.array2d(dtype=wp.float64),
    velocities: wp.array2d(dtype=wp.float64),
    cell_item: int,
    coordinate_item: int,
) -> Vec6d:
    base_row = cell_item * SPATIAL_DIM
    value0 = wp.float64(0.0)
    value1 = wp.float64(0.0)
    value2 = wp.float64(0.0)
    value3 = wp.float64(0.0)
    value4 = wp.float64(0.0)
    value5 = wp.float64(0.0)
    for column in range(MAX_DOF):
        velocity = velocities[coordinate_item, column]
        value0 += basis[base_row + 0, column] * velocity
        value1 += basis[base_row + 1, column] * velocity
        value2 += basis[base_row + 2, column] * velocity
        value3 += basis[base_row + 3, column] * velocity
        value4 += basis[base_row + 4, column] * velocity
        value5 += basis[base_row + 5, column] * velocity
    return Vec6d(value0, value1, value2, value3, value4, value5)


@wp.func
def _left_coefficients(angle_sq: wp.float64) -> wp.vec4d:
    c1 = wp.float64(0.0)
    c2 = wp.float64(0.0)
    c3 = wp.float64(0.0)
    c4 = wp.float64(0.0)
    if angle_sq <= wp.float64(0.00580466633864119):
        x = angle_sq
        c1 = wp.float64(0.5) + x * x * (
            -wp.float64(1.0 / 720.0)
            + x
            * (
                wp.float64(1.0 / 20160.0)
                - x * wp.float64(1.0 / 1209600.0)
            )
        )
        c2 = wp.float64(1.0 / 6.0) + x * x * (
            -wp.float64(1.0 / 5040.0)
            + x
            * (
                wp.float64(1.0 / 181440.0)
                - x * wp.float64(1.0 / 13305600.0)
            )
        )
        c3 = wp.float64(1.0 / 24.0) + x * (
            -wp.float64(1.0 / 360.0)
            + x
            * (
                wp.float64(1.0 / 13440.0)
                + x
                * (
                    -wp.float64(1.0 / 907200.0)
                    + x * wp.float64(1.0 / 95800320.0)
                )
            )
        )
        c4 = wp.float64(1.0 / 120.0) + x * (
            -wp.float64(1.0 / 2520.0)
            + x
            * (
                wp.float64(1.0 / 120960.0)
                + x
                * (
                    -wp.float64(1.0 / 9979200.0)
                    + x * wp.float64(1.0 / 1245404160.0)
                )
            )
        )
    else:
        theta = wp.sqrt(angle_sq)
        sine = wp.sin(theta)
        cosine = wp.cos(theta)
        theta2 = theta * theta
        theta3 = theta2 * theta
        theta4 = theta2 * theta2
        theta5 = theta4 * theta
        c1 = (wp.float64(4.0) - wp.float64(4.0) * cosine - theta * sine) / (
            wp.float64(2.0) * theta2
        )
        c2 = (
            wp.float64(4.0) * theta
            - wp.float64(5.0) * sine
            + theta * cosine
        ) / (wp.float64(2.0) * theta3)
        c3 = (wp.float64(2.0) - wp.float64(2.0) * cosine - theta * sine) / (
            wp.float64(2.0) * theta4
        )
        c4 = (
            wp.float64(2.0) * theta
            - wp.float64(3.0) * sine
            + theta * cosine
        ) / (wp.float64(2.0) * theta5)
    return wp.vec4d(c1, c2, c3, c4)


@wp.func
def _left_coefficient_x_derivatives(angle_sq: wp.float64) -> wp.vec4d:
    d1 = wp.float64(0.0)
    d2 = wp.float64(0.0)
    d3 = wp.float64(0.0)
    d4 = wp.float64(0.0)
    if angle_sq <= wp.float64(0.00580466633864119):
        x = angle_sq
        d1 = x * (
            -wp.float64(1.0 / 360.0)
            + x
            * (
                wp.float64(1.0 / 6720.0)
                - x * wp.float64(1.0 / 302400.0)
            )
        )
        d2 = x * (
            -wp.float64(1.0 / 2520.0)
            + x
            * (
                wp.float64(1.0 / 60480.0)
                - x * wp.float64(1.0 / 3326400.0)
            )
        )
        d3 = -wp.float64(1.0 / 360.0) + x * (
            wp.float64(1.0 / 6720.0)
            + x
            * (
                -wp.float64(1.0 / 302400.0)
                + x * wp.float64(1.0 / 23950080.0)
            )
        )
        d4 = -wp.float64(1.0 / 2520.0) + x * (
            wp.float64(1.0 / 60480.0)
            + x
            * (
                -wp.float64(1.0 / 3326400.0)
                + x * wp.float64(1.0 / 311351040.0)
            )
        )
    else:
        theta = wp.sqrt(angle_sq)
        sine = wp.sin(theta)
        cosine = wp.cos(theta)
        theta2 = theta * theta
        theta3 = theta2 * theta
        theta4 = theta2 * theta2
        theta5 = theta4 * theta
        theta6 = theta3 * theta3
        theta7 = theta6 * theta
        common = (
            -wp.float64(8.0)
            + (wp.float64(8.0) - theta2) * cosine
            + wp.float64(5.0) * theta * sine
        )
        alternate = (
            -wp.float64(8.0) * theta
            + (wp.float64(15.0) - theta2) * sine
            - wp.float64(7.0) * theta * cosine
        )
        d1 = common / (wp.float64(4.0) * theta4)
        d2 = alternate / (wp.float64(4.0) * theta5)
        d3 = common / (wp.float64(4.0) * theta6)
        d4 = alternate / (wp.float64(4.0) * theta7)
    return wp.vec4d(d1, d2, d3, d4)


@wp.func
def _left_action(xi: Vec6d, value: Vec6d, coefficients: wp.vec4d) -> Vec6d:
    result = value
    power = value
    for order in range(4):
        power = _ad_action(xi, power)
        result += coefficients[order] * power
    return result


@wp.func
def _left_total_derivative_action(
    xi: Vec6d,
    xi_dot: Vec6d,
    value: Vec6d,
    value_dot: Vec6d,
    coefficients: wp.vec4d,
    coefficient_directions: wp.vec4d,
) -> Vec6d:
    result = value_dot
    power = value
    power_dot = value_dot
    for order in range(4):
        power_previous = power
        power = _ad_action(xi, power_previous)
        power_dot = _ad_action(xi_dot, power_previous) + _ad_action(
            xi, power_dot
        )
        result += coefficient_directions[order] * power
        result += coefficients[order] * power_dot
    return result


@wp.func
def _forward_coefficients(angle_sq: wp.float64) -> wp.vec3d:
    sinc = wp.float64(0.0)
    cosc = wp.float64(0.0)
    tanc = wp.float64(0.0)
    if angle_sq <= wp.float64(0.0024607823917256556):
        x = angle_sq
        sinc = wp.float64(1.0) + x * (
            -wp.float64(1.0 / 6.0)
            + x
            * (
                wp.float64(1.0 / 120.0)
                + x
                * (
                    -wp.float64(1.0 / 5040.0)
                    + x * wp.float64(1.0 / 362880.0)
                )
            )
        )
        cosc = wp.float64(0.5) + x * (
            -wp.float64(1.0 / 24.0)
            + x
            * (
                wp.float64(1.0 / 720.0)
                + x
                * (
                    -wp.float64(1.0 / 40320.0)
                    + x * wp.float64(1.0 / 3628800.0)
                )
            )
        )
        tanc = wp.float64(1.0 / 6.0) + x * (
            -wp.float64(1.0 / 120.0)
            + x
            * (
                wp.float64(1.0 / 5040.0)
                + x
                * (
                    -wp.float64(1.0 / 362880.0)
                    + x * wp.float64(1.0 / 39916800.0)
                )
            )
        )
    else:
        theta = wp.sqrt(angle_sq)
        sinc = wp.sin(theta) / theta
        cosc = (wp.float64(1.0) - wp.cos(theta)) / angle_sq
        tanc = (theta - wp.sin(theta)) / (angle_sq * theta)
    return wp.vec3d(sinc, cosc, tanc)


@wp.func
def _rotation_entry(
    omega: wp.vec3d,
    angle_sq: wp.float64,
    forward: wp.vec3d,
    row: int,
    column: int,
) -> wp.float64:
    delta = wp.float64(0.0)
    if row == column:
        delta = wp.float64(1.0)
    skew = wp.float64(0.0)
    if row == 0 and column == 1:
        skew = -omega[2]
    elif row == 0 and column == 2:
        skew = omega[1]
    elif row == 1 and column == 0:
        skew = omega[2]
    elif row == 1 and column == 2:
        skew = -omega[0]
    elif row == 2 and column == 0:
        skew = -omega[1]
    elif row == 2 and column == 1:
        skew = omega[0]
    square = omega[row] * omega[column] - angle_sq * delta
    return delta + forward[0] * skew + forward[1] * square


@wp.func
def _translation(
    omega: wp.vec3d, linear: wp.vec3d, forward: wp.vec3d
) -> wp.vec3d:
    first = wp.cross(omega, linear)
    second = wp.cross(omega, first)
    return linear + forward[1] * first + forward[2] * second


@wp.func
def _adjoint_inverse_entry(
    omega: wp.vec3d,
    translation: wp.vec3d,
    angle_sq: wp.float64,
    forward: wp.vec3d,
    row: int,
    column: int,
) -> wp.float64:
    value = wp.float64(0.0)
    if row < 3 and column < 3:
        value = _rotation_entry(omega, angle_sq, forward, column, row)
    elif row >= 3 and column >= 3:
        value = _rotation_entry(
            omega, angle_sq, forward, column - 3, row - 3
        )
    elif row >= 3 and column < 3:
        local_row = row - 3
        for k in range(3):
            skew = wp.float64(0.0)
            if k == 0 and column == 1:
                skew = -translation[2]
            elif k == 0 and column == 2:
                skew = translation[1]
            elif k == 1 and column == 0:
                skew = translation[2]
            elif k == 1 and column == 2:
                skew = -translation[0]
            elif k == 2 and column == 0:
                skew = -translation[1]
            elif k == 2 and column == 1:
                skew = translation[0]
            value -= _rotation_entry(
                omega, angle_sq, forward, k, local_row
            ) * skew
    return value


@wp.func
def _adjoint_inverse_action(
    omega: wp.vec3d,
    translation: wp.vec3d,
    angle_sq: wp.float64,
    forward: wp.vec3d,
    value: Vec6d,
) -> Vec6d:
    result = Vec6d()
    for row in range(SPATIAL_DIM):
        row_value = wp.float64(0.0)
        for column in range(SPATIAL_DIM):
            row_value += _adjoint_inverse_entry(
                omega,
                translation,
                angle_sq,
                forward,
                row,
                column,
            ) * value[column]
        result[row] = row_value
    return result


@wp.kernel
def matrix_free_cell_terms_kernel(
    q_link: wp.array2d(dtype=wp.float64),
    qd_link: wp.array2d(dtype=wp.float64),
    basis_z1: wp.array2d(dtype=wp.float64),
    basis_z2: wp.array2d(dtype=wp.float64),
    reference_z1: wp.array2d(dtype=wp.float64),
    reference_z2: wp.array2d(dtype=wp.float64),
    segment_lengths: wp.array(dtype=wp.float64),
    cell_widths: wp.array(dtype=wp.float64),
    gather_indices: wp.array2d(dtype=wp.int32),
    gather_mask: wp.array2d(dtype=wp.float64),
    adjoint: wp.array2d(dtype=wp.float64),
    tangent_active: wp.array2d(dtype=wp.float64),
    link_velocity: wp.array2d(dtype=wp.float64),
    step_velocity: wp.array2d(dtype=wp.float64),
    tangent_velocity_dot: wp.array2d(dtype=wp.float64),
):
    """Compute one active tangent entry per thread and vectors per row leader."""

    linear_index = wp.tid()
    entries_per_cell = SPATIAL_DIM * NUM_DOFS
    work_item = linear_index // entries_per_cell
    entry = linear_index - work_item * entries_per_cell
    row = entry // NUM_DOFS
    column = entry - row * NUM_DOFS
    cells_per_environment = NUM_SEGMENTS * NUM_CELLS
    environment = work_item // cells_per_environment
    environment_cell = work_item - environment * cells_per_environment
    segment = environment_cell // NUM_CELLS
    cell = environment_cell - segment * NUM_CELLS
    cell_item = segment * NUM_CELLS + cell
    coordinate_item = environment * NUM_SEGMENTS + segment
    base_row = cell_item * SPATIAL_DIM

    xi1 = _strain(
        basis_z1, reference_z1, q_link, cell_item, coordinate_item
    )
    xi2 = _strain(
        basis_z2, reference_z2, q_link, cell_item, coordinate_item
    )
    xid1 = _strain_rate(basis_z1, qd_link, cell_item, coordinate_item)
    xid2 = _strain_rate(basis_z2, qd_link, cell_item, coordinate_item)
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

    active_value = wp.float64(0.0)
    for local_column in range(MAX_DOF):
        if gather_indices[segment, local_column] == column:
            basis1 = _load_column(basis_z1, base_row, local_column)
            basis2 = _load_column(basis_z2, base_row, local_column)
            magnus_basis = alpha * (basis1 + basis2) + commutator_coefficient * (
                _ad_action(xi1, basis2) - _ad_action(xi2, basis1)
            )
            local_tangent = _left_action(magnus, magnus_basis, coefficients)
            active_value += (
                local_tangent[row] * gather_mask[segment, local_column]
            )
    tangent_active[work_item * SPATIAL_DIM + row, column] = active_value

    omega = wp.vec3d(magnus[0], magnus[1], magnus[2])
    linear = wp.vec3d(magnus[3], magnus[4], magnus[5])
    forward = _forward_coefficients(angle_sq)
    translation = _translation(omega, linear, forward)
    if column < SPATIAL_DIM:
        adjoint[work_item * SPATIAL_DIM + row, column] = (
            _adjoint_inverse_entry(
                omega,
                translation,
                angle_sq,
                forward,
                row,
                column,
            )
        )

    if column == 0:
        link = _left_action(magnus, magnus_dot, coefficients)
        step = _adjoint_inverse_action(
            omega, translation, angle_sq, forward, link
        )
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
        coefficient_directions = coefficient_derivatives * angle_sq_dot
        tangent_dot_velocity = _left_total_derivative_action(
            magnus,
            magnus_dot,
            magnus_dot,
            magnus_basis_dot_qd,
            coefficients,
            coefficient_directions,
        )
        vector_row = work_item * SPATIAL_DIM + row
        link_velocity[vector_row, 0] = link[row]
        step_velocity[vector_row, 0] = step[row]
        tangent_velocity_dot[vector_row, 0] = tangent_dot_velocity[row]


def _matrix_free_cell_terms(
    q_link: wp.array2d(dtype=wp.float64),
    qd_link: wp.array2d(dtype=wp.float64),
    basis_z1: wp.array2d(dtype=wp.float64),
    basis_z2: wp.array2d(dtype=wp.float64),
    reference_z1: wp.array2d(dtype=wp.float64),
    reference_z2: wp.array2d(dtype=wp.float64),
    segment_lengths: wp.array(dtype=wp.float64),
    cell_widths: wp.array(dtype=wp.float64),
    gather_indices: wp.array2d(dtype=wp.int32),
    gather_mask: wp.array2d(dtype=wp.float64),
    adjoint: wp.array2d(dtype=wp.float64),
    tangent_active: wp.array2d(dtype=wp.float64),
    link_velocity: wp.array2d(dtype=wp.float64),
    step_velocity: wp.array2d(dtype=wp.float64),
    tangent_velocity_dot: wp.array2d(dtype=wp.float64),
):
    work_items = q_link.shape[0] * NUM_CELLS
    wp.launch(
        matrix_free_cell_terms_kernel,
        dim=work_items * SPATIAL_DIM * NUM_DOFS,
        inputs=[
            q_link,
            qd_link,
            basis_z1,
            basis_z2,
            reference_z1,
            reference_z2,
            segment_lengths,
            cell_widths,
            gather_indices,
            gather_mask,
        ],
        outputs=[
            adjoint,
            tangent_active,
            link_velocity,
            step_velocity,
            tangent_velocity_dot,
        ],
        block_dim=128,
    )


matrix_free_cell_terms = wp.jax_callable(_matrix_free_cell_terms, num_outputs=5)


@wp.kernel
def matrix_free_cell_terms_serial_kernel(
    q_link: wp.array2d(dtype=wp.float64),
    qd_link: wp.array2d(dtype=wp.float64),
    basis_z1: wp.array2d(dtype=wp.float64),
    basis_z2: wp.array2d(dtype=wp.float64),
    reference_z1: wp.array2d(dtype=wp.float64),
    reference_z2: wp.array2d(dtype=wp.float64),
    segment_lengths: wp.array(dtype=wp.float64),
    cell_widths: wp.array(dtype=wp.float64),
    gather_indices: wp.array2d(dtype=wp.int32),
    gather_mask: wp.array2d(dtype=wp.float64),
    adjoint: wp.array2d(dtype=wp.float64),
    tangent_active: wp.array2d(dtype=wp.float64),
    link_velocity: wp.array2d(dtype=wp.float64),
    step_velocity: wp.array2d(dtype=wp.float64),
    tangent_velocity_dot: wp.array2d(dtype=wp.float64),
):
    """Compute each cell's shared Lie data once in one owning thread."""

    work_item = wp.tid()
    cells_per_environment = NUM_SEGMENTS * NUM_CELLS
    environment = work_item // cells_per_environment
    environment_cell = work_item - environment * cells_per_environment
    segment = environment_cell // NUM_CELLS
    cell = environment_cell - segment * NUM_CELLS
    cell_item = segment * NUM_CELLS + cell
    coordinate_item = environment * NUM_SEGMENTS + segment
    base_row = cell_item * SPATIAL_DIM
    output_base_row = work_item * SPATIAL_DIM

    xi1 = _strain(
        basis_z1, reference_z1, q_link, cell_item, coordinate_item
    )
    xi2 = _strain(
        basis_z2, reference_z2, q_link, cell_item, coordinate_item
    )
    xid1 = _strain_rate(basis_z1, qd_link, cell_item, coordinate_item)
    xid2 = _strain_rate(basis_z2, qd_link, cell_item, coordinate_item)
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

    for row in range(SPATIAL_DIM):
        for column in range(NUM_DOFS):
            tangent_active[output_base_row + row, column] = wp.float64(0.0)
        for column in range(SPATIAL_DIM):
            adjoint[output_base_row + row, column] = _adjoint_inverse_entry(
                omega,
                translation,
                angle_sq,
                forward,
                row,
                column,
            )

    for local_column in range(MAX_DOF):
        basis1 = _load_column(basis_z1, base_row, local_column)
        basis2 = _load_column(basis_z2, base_row, local_column)
        magnus_basis = alpha * (basis1 + basis2) + commutator_coefficient * (
            _ad_action(xi1, basis2) - _ad_action(xi2, basis1)
        )
        local_tangent = _left_action(magnus, magnus_basis, coefficients)
        active_column = gather_indices[segment, local_column]
        mask = gather_mask[segment, local_column]
        for row in range(SPATIAL_DIM):
            tangent_active[output_base_row + row, active_column] += (
                local_tangent[row] * mask
            )

    link = _left_action(magnus, magnus_dot, coefficients)
    step = _adjoint_inverse_action(
        omega, translation, angle_sq, forward, link
    )
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
    tangent_dot_velocity = _left_total_derivative_action(
        magnus,
        magnus_dot,
        magnus_dot,
        magnus_basis_dot_qd,
        coefficients,
        coefficient_derivatives * angle_sq_dot,
    )
    for row in range(SPATIAL_DIM):
        link_velocity[output_base_row + row, 0] = link[row]
        step_velocity[output_base_row + row, 0] = step[row]
        tangent_velocity_dot[output_base_row + row, 0] = (
            tangent_dot_velocity[row]
        )


def _matrix_free_cell_terms_serial(
    q_link: wp.array2d(dtype=wp.float64),
    qd_link: wp.array2d(dtype=wp.float64),
    basis_z1: wp.array2d(dtype=wp.float64),
    basis_z2: wp.array2d(dtype=wp.float64),
    reference_z1: wp.array2d(dtype=wp.float64),
    reference_z2: wp.array2d(dtype=wp.float64),
    segment_lengths: wp.array(dtype=wp.float64),
    cell_widths: wp.array(dtype=wp.float64),
    gather_indices: wp.array2d(dtype=wp.int32),
    gather_mask: wp.array2d(dtype=wp.float64),
    adjoint: wp.array2d(dtype=wp.float64),
    tangent_active: wp.array2d(dtype=wp.float64),
    link_velocity: wp.array2d(dtype=wp.float64),
    step_velocity: wp.array2d(dtype=wp.float64),
    tangent_velocity_dot: wp.array2d(dtype=wp.float64),
):
    work_items = q_link.shape[0] * NUM_CELLS
    wp.launch(
        matrix_free_cell_terms_serial_kernel,
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
            gather_indices,
            gather_mask,
        ],
        outputs=[
            adjoint,
            tangent_active,
            link_velocity,
            step_velocity,
            tangent_velocity_dot,
        ],
        block_dim=128,
    )


matrix_free_cell_terms_serial = wp.jax_callable(
    _matrix_free_cell_terms_serial, num_outputs=5
)
