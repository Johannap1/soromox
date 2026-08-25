# ruff: noqa: I001, UP018
"""Exact specializations for the fixed order-zero GVS benchmark."""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})

from tools.benchmarks.gvs_warp_kernels import (
    MAX_DOF,
    NUM_CELLS,
    NUM_DOFS,
    NUM_SEGMENTS,
    SPATIAL_DIM,
)
from tools.benchmarks.gvs_warp_lie_kernels import (
    Vec6d,
    _adjoint_inverse_action,
    _adjoint_inverse_entry,
    _forward_coefficients,
    _left_action,
    _left_coefficient_x_derivatives,
    _left_coefficients,
    _left_total_derivative_action,
    _load_column,
    _strain,
    _strain_rate,
    _translation,
)


@wp.kernel(enable_backward=False)
def constant_strain_cell_terms_kernel(
    q_link: wp.array2d(dtype=wp.float64),
    qd_link: wp.array2d(dtype=wp.float64),
    basis: wp.array2d(dtype=wp.float64),
    reference: wp.array2d(dtype=wp.float64),
    segment_lengths: wp.array(dtype=wp.float64),
    cell_widths: wp.array(dtype=wp.float64),
    adjoint: wp.array2d(dtype=wp.float64),
    tangent_active: wp.array2d(dtype=wp.float64),
    link_velocity: wp.array2d(dtype=wp.float64),
    step_velocity: wp.array2d(dtype=wp.float64),
    tangent_velocity_dot: wp.array2d(dtype=wp.float64),
):
    """Exploit B_Z1 == B_Z2 exactly, with one owning thread per cell."""

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

    xi = _strain(basis, reference, q_link, cell_item, coordinate_item)
    xi_dot = _strain_rate(basis, qd_link, cell_item, coordinate_item)
    scale = segment_lengths[segment] * cell_widths[cell_item]
    magnus = scale * xi
    magnus_dot = scale * xi_dot
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
        magnus_basis = scale * _load_column(basis, base_row, local_column)
        local_tangent = _left_action(magnus, magnus_basis, coefficients)
        active_column = segment * MAX_DOF + local_column
        for row in range(SPATIAL_DIM):
            tangent_active[output_base_row + row, active_column] = local_tangent[row]

    link = _left_action(magnus, magnus_dot, coefficients)
    step = _adjoint_inverse_action(
        omega, translation, angle_sq, forward, link
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
        Vec6d(),
        coefficients,
        coefficient_derivatives * angle_sq_dot,
    )
    for row in range(SPATIAL_DIM):
        link_velocity[output_base_row + row, 0] = link[row]
        step_velocity[output_base_row + row, 0] = step[row]
        tangent_velocity_dot[output_base_row + row, 0] = (
            tangent_dot_velocity[row]
        )


def _constant_strain_cell_terms(
    q_link: wp.array2d(dtype=wp.float64),
    qd_link: wp.array2d(dtype=wp.float64),
    basis: wp.array2d(dtype=wp.float64),
    reference: wp.array2d(dtype=wp.float64),
    segment_lengths: wp.array(dtype=wp.float64),
    cell_widths: wp.array(dtype=wp.float64),
    adjoint: wp.array2d(dtype=wp.float64),
    tangent_active: wp.array2d(dtype=wp.float64),
    link_velocity: wp.array2d(dtype=wp.float64),
    step_velocity: wp.array2d(dtype=wp.float64),
    tangent_velocity_dot: wp.array2d(dtype=wp.float64),
):
    work_items = q_link.shape[0] * NUM_CELLS
    wp.launch(
        constant_strain_cell_terms_kernel,
        dim=work_items,
        inputs=[
            q_link,
            qd_link,
            basis,
            reference,
            segment_lengths,
            cell_widths,
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


constant_strain_cell_terms = wp.jax_callable(
    _constant_strain_cell_terms, num_outputs=5
)
