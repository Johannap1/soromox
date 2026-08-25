# ruff: noqa: I001, UP018
"""Shape-generic forward-only GVS joint kernels.

The GVS joint basis already encodes the concrete joint type.  This module
therefore evaluates every supported joint through the same runtime-sized
SE(3) path instead of generating a kernel for each joint type or model shape.
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
def _joint_strain(
    basis: wp.array2d(dtype=wp.float64),
    reference: wp.array2d(dtype=wp.float64),
    coordinates: wp.array2d(dtype=wp.float64),
    local_to_global: wp.array2d(dtype=wp.int32),
    environment: int,
    segment: int,
) -> Vec6d:
    value = Vec6d(
        reference[segment, 0],
        reference[segment, 1],
        reference[segment, 2],
        reference[segment, 3],
        reference[segment, 4],
        reference[segment, 5],
    )
    base_row = segment * SPATIAL_DIM
    local_column = int(0)
    while local_column < local_to_global.shape[1]:
        global_column = local_to_global[segment, local_column]
        if global_column >= 0:
            coordinate = coordinates[environment, global_column]
            row = int(0)
            while row < SPATIAL_DIM:
                value[row] += basis[base_row + row, local_column] * coordinate
                row += 1
        local_column += 1
    return value


@wp.func
def _joint_strain_rate(
    basis: wp.array2d(dtype=wp.float64),
    velocities: wp.array2d(dtype=wp.float64),
    local_to_global: wp.array2d(dtype=wp.int32),
    environment: int,
    segment: int,
) -> Vec6d:
    value = Vec6d()
    base_row = segment * SPATIAL_DIM
    local_column = int(0)
    while local_column < local_to_global.shape[1]:
        global_column = local_to_global[segment, local_column]
        if global_column >= 0:
            velocity = velocities[environment, global_column]
            row = int(0)
            while row < SPATIAL_DIM:
                value[row] += basis[base_row + row, local_column] * velocity
                row += 1
        local_column += 1
    return value


@wp.func
def _joint_basis_column(
    basis: wp.array2d(dtype=wp.float64), segment: int, column: int
) -> Vec6d:
    base_row = segment * SPATIAL_DIM
    return Vec6d(
        basis[base_row + 0, column],
        basis[base_row + 1, column],
        basis[base_row + 2, column],
        basis[base_row + 3, column],
        basis[base_row + 4, column],
        basis[base_row + 5, column],
    )


@wp.kernel(enable_backward=False)
def scalable_joint_terms_kernel(
    q: wp.array2d(dtype=wp.float64),
    qd: wp.array2d(dtype=wp.float64),
    basis: wp.array2d(dtype=wp.float64),
    reference: wp.array2d(dtype=wp.float64),
    local_to_global: wp.array2d(dtype=wp.int32),
    adjoint: wp.array2d(dtype=wp.float64),
    adjoint_dot: wp.array2d(dtype=wp.float64),
    tangent_local: wp.array2d(dtype=wp.float64),
    tangent_dot_qd: wp.array2d(dtype=wp.float64),
    joint_velocity: wp.array2d(dtype=wp.float64),
):
    """Evaluate one joint per ``(environment, segment)`` work item."""

    work_item = wp.tid()
    num_segments = reference.shape[0]
    environment = work_item // num_segments
    segment = work_item - environment * num_segments
    output_base_row = work_item * SPATIAL_DIM

    xi = _joint_strain(
        basis, reference, q, local_to_global, environment, segment
    )
    xid = _joint_strain_rate(
        basis, qd, local_to_global, environment, segment
    )
    angle_sq = xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2]
    coefficients = _left_coefficients(angle_sq)
    coefficient_derivatives = _left_coefficient_x_derivatives(angle_sq)
    angle_sq_dot = wp.float64(2.0) * (
        xi[0] * xid[0] + xi[1] * xid[1] + xi[2] * xid[2]
    )
    omega = wp.vec3d(xi[0], xi[1], xi[2])
    linear = wp.vec3d(xi[3], xi[4], xi[5])
    forward = _forward_coefficients(angle_sq)
    translation = _translation(omega, linear, forward)

    velocity = _left_action(xi, xid, coefficients)
    eta = _adjoint_inverse_action(
        omega, translation, angle_sq, forward, velocity
    )
    derivative_velocity = _left_total_derivative_action(
        xi,
        xid,
        xid,
        Vec6d(),
        coefficients,
        coefficient_derivatives * angle_sq_dot,
    )

    row = int(0)
    while row < SPATIAL_DIM:
        tangent_dot_qd[output_base_row + row, 0] = derivative_velocity[row]
        joint_velocity[output_base_row + row, 0] = velocity[row]
        column = int(0)
        while column < SPATIAL_DIM:
            adjoint_value = _adjoint_inverse_entry(
                omega,
                translation,
                angle_sq,
                forward,
                row,
                column,
            )
            adjoint[output_base_row + row, column] = adjoint_value
            column += 1
        row += 1

    # Ad_inv_dot = -ad(Ad_inv * joint_velocity) * Ad_inv.  Applying ad to
    # each already-computed adjoint column avoids materializing a dense ad.
    column = int(0)
    while column < SPATIAL_DIM:
        adjoint_column = Vec6d()
        row = int(0)
        while row < SPATIAL_DIM:
            adjoint_column[row] = adjoint[output_base_row + row, column]
            row += 1
        derivative_column = -_ad_action(eta, adjoint_column)
        row = int(0)
        while row < SPATIAL_DIM:
            adjoint_dot[output_base_row + row, column] = derivative_column[row]
            row += 1
        column += 1

    local_column = int(0)
    while local_column < local_to_global.shape[1]:
        tangent = _left_action(
            xi,
            _joint_basis_column(basis, segment, local_column),
            coefficients,
        )
        row = int(0)
        while row < SPATIAL_DIM:
            tangent_local[output_base_row + row, local_column] = tangent[row]
            row += 1
        local_column += 1


def _scalable_joint_terms(
    q: wp.array2d(dtype=wp.float64),
    qd: wp.array2d(dtype=wp.float64),
    basis: wp.array2d(dtype=wp.float64),
    reference: wp.array2d(dtype=wp.float64),
    local_to_global: wp.array2d(dtype=wp.int32),
    adjoint: wp.array2d(dtype=wp.float64),
    adjoint_dot: wp.array2d(dtype=wp.float64),
    tangent_local: wp.array2d(dtype=wp.float64),
    tangent_dot_qd: wp.array2d(dtype=wp.float64),
    joint_velocity: wp.array2d(dtype=wp.float64),
):
    wp.launch(
        scalable_joint_terms_kernel,
        dim=q.shape[0] * reference.shape[0],
        inputs=[q, qd, basis, reference, local_to_global],
        outputs=[
            adjoint,
            adjoint_dot,
            tangent_local,
            tangent_dot_qd,
            joint_velocity,
        ],
        block_dim=128,
    )


scalable_joint_terms = wp.jax_callable(_scalable_joint_terms, num_outputs=5)
