# ruff: noqa: I001, UP018
"""Runtime-sized forward-only GVS dynamics assembly kernels."""

from __future__ import annotations

import warp as wp

from tools.benchmarks.gvs_warp_lie_kernels import Vec6d

wp.set_module_options({"enable_backward": False})


SPATIAL_DIM = 6


@wp.func
def _state_row(
    environment: int,
    segment: int,
    quadrature: int,
    row: int,
    num_segments: int,
    num_quadrature: int,
) -> int:
    return (
        ((environment * num_segments + segment) * num_quadrature + quadrature)
        * SPATIAL_DIM
        + row
    )


@wp.kernel(enable_backward=False)
def scalable_inertia_assembly_kernel(
    jacobians: wp.array2d(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    mass_diagonals: wp.array2d(dtype=wp.float64),
    num_quadrature_array: wp.array(dtype=wp.int32),
    inertia: wp.array3d(dtype=wp.float64),
):
    """Assign one unique thread to each batched mass-matrix entry."""

    linear_index = wp.tid()
    num_dofs = inertia.shape[1]
    entries_per_environment = num_dofs * num_dofs
    environment = linear_index // entries_per_environment
    entry = linear_index - environment * entries_per_environment
    output_row = entry // num_dofs
    output_column = entry - output_row * num_dofs
    num_quadrature = num_quadrature_array[0]
    num_segments = weights.shape[0] // num_quadrature

    value = wp.float64(0.0)
    segment = int(0)
    while segment < num_segments:
        quadrature = int(0)
        while quadrature < num_quadrature:
            quadrature_item = segment * num_quadrature + quadrature
            weight = weights[quadrature_item]
            row = int(0)
            while row < SPATIAL_DIM:
                state_row = _state_row(
                    environment,
                    segment,
                    quadrature,
                    row,
                    num_segments,
                    num_quadrature,
                )
                value += (
                    jacobians[state_row, output_row]
                    * weight
                    * mass_diagonals[quadrature_item, row]
                    * jacobians[state_row, output_column]
                )
                row += 1
            quadrature += 1
        segment += 1
    inertia[environment, output_row, output_column] = value


@wp.func
def _coadjoint_wrench(
    velocity: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd: wp.array2d(dtype=wp.float64),
    mass_diagonals: wp.array2d(dtype=wp.float64),
    state_base_row: int,
    quadrature_item: int,
) -> Vec6d:
    omega = wp.vec3d(
        velocity[state_base_row + 0, 0],
        velocity[state_base_row + 1, 0],
        velocity[state_base_row + 2, 0],
    )
    linear_velocity = wp.vec3d(
        velocity[state_base_row + 3, 0],
        velocity[state_base_row + 4, 0],
        velocity[state_base_row + 5, 0],
    )
    moment = wp.vec3d(
        mass_diagonals[quadrature_item, 0] * omega[0],
        mass_diagonals[quadrature_item, 1] * omega[1],
        mass_diagonals[quadrature_item, 2] * omega[2],
    )
    force = wp.vec3d(
        mass_diagonals[quadrature_item, 3] * linear_velocity[0],
        mass_diagonals[quadrature_item, 4] * linear_velocity[1],
        mass_diagonals[quadrature_item, 5] * linear_velocity[2],
    )
    angular = wp.cross(omega, moment) + wp.cross(linear_velocity, force)
    linear = wp.cross(omega, force)
    return Vec6d(
        mass_diagonals[quadrature_item, 0]
        * jacobian_dot_qd[state_base_row + 0, 0]
        + angular[0],
        mass_diagonals[quadrature_item, 1]
        * jacobian_dot_qd[state_base_row + 1, 0]
        + angular[1],
        mass_diagonals[quadrature_item, 2]
        * jacobian_dot_qd[state_base_row + 2, 0]
        + angular[2],
        mass_diagonals[quadrature_item, 3]
        * jacobian_dot_qd[state_base_row + 3, 0]
        + linear[0],
        mass_diagonals[quadrature_item, 4]
        * jacobian_dot_qd[state_base_row + 4, 0]
        + linear[1],
        mass_diagonals[quadrature_item, 5]
        * jacobian_dot_qd[state_base_row + 5, 0]
        + linear[2],
    )


@wp.kernel(enable_backward=False)
def scalable_force_assembly_kernel(
    jacobians: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd: wp.array2d(dtype=wp.float64),
    velocity: wp.array2d(dtype=wp.float64),
    gravity: wp.array2d(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    mass_diagonals: wp.array2d(dtype=wp.float64),
    num_quadrature_array: wp.array(dtype=wp.int32),
    coriolis_qd: wp.array2d(dtype=wp.float64),
    gravity_force: wp.array2d(dtype=wp.float64),
):
    """Assign one unique thread to each batched generalized-force entry."""

    linear_index = wp.tid()
    num_dofs = coriolis_qd.shape[1]
    environment = linear_index // num_dofs
    output_column = linear_index - environment * num_dofs
    num_quadrature = num_quadrature_array[0]
    num_segments = weights.shape[0] // num_quadrature

    coriolis_value = wp.float64(0.0)
    gravity_value = wp.float64(0.0)
    segment = int(0)
    while segment < num_segments:
        quadrature = int(0)
        while quadrature < num_quadrature:
            quadrature_item = segment * num_quadrature + quadrature
            state_base_row = _state_row(
                environment,
                segment,
                quadrature,
                0,
                num_segments,
                num_quadrature,
            )
            wrench = _coadjoint_wrench(
                velocity,
                jacobian_dot_qd,
                mass_diagonals,
                state_base_row,
                quadrature_item,
            )
            weight = weights[quadrature_item]
            row = int(0)
            while row < SPATIAL_DIM:
                jacobian_value = jacobians[state_base_row + row, output_column]
                coriolis_value += weight * jacobian_value * wrench[row]
                gravity_value -= (
                    weight
                    * jacobian_value
                    * mass_diagonals[quadrature_item, row]
                    * gravity[state_base_row + row, 0]
                )
                row += 1
            quadrature += 1
        segment += 1

    coriolis_qd[environment, output_column] = coriolis_value
    gravity_force[environment, output_column] = gravity_value


def _scalable_assemble_dynamics(
    jacobians: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd: wp.array2d(dtype=wp.float64),
    velocity: wp.array2d(dtype=wp.float64),
    gravity: wp.array2d(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    mass_diagonals: wp.array2d(dtype=wp.float64),
    num_quadrature: wp.array(dtype=wp.int32),
    inertia: wp.array3d(dtype=wp.float64),
    coriolis_qd: wp.array2d(dtype=wp.float64),
    gravity_force: wp.array2d(dtype=wp.float64),
):
    batch_size = coriolis_qd.shape[0]
    num_dofs = coriolis_qd.shape[1]
    wp.launch(
        scalable_inertia_assembly_kernel,
        dim=batch_size * num_dofs * num_dofs,
        inputs=[jacobians, weights, mass_diagonals, num_quadrature],
        outputs=[inertia],
        block_dim=128,
    )
    wp.launch(
        scalable_force_assembly_kernel,
        dim=batch_size * num_dofs,
        inputs=[
            jacobians,
            jacobian_dot_qd,
            velocity,
            gravity,
            weights,
            mass_diagonals,
            num_quadrature,
        ],
        outputs=[coriolis_qd, gravity_force],
        block_dim=128,
    )


scalable_assemble_dynamics = wp.jax_callable(
    _scalable_assemble_dynamics, num_outputs=3
)
