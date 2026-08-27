"""Family executor for PlanarPCS and PCS dynamics in Warp."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from soromox.systems._execution.warp.pcs.operands import PCSOperands
from soromox.systems._execution.warp.pcs.planar_kernels import (
    planar_local_operators,
    planar_persistent_chain,
)
from soromox.systems._execution.warp.pcs.spatial_kernels import (
    spatial_local_operators,
    spatial_persistent_chain,
)


def _planar_dynamics_terms(
    operands: PCSOperands, q: Array, qd: Array
) -> tuple[Array, Array, Array]:
    """Execute the staged constant-strain SE(2) pipeline."""

    batch_size = q.shape[0]
    num_segments = operands.num_segments
    num_dofs = operands.num_dofs
    points_per_segment = operands.num_gauss_points + 1
    operator_rows = batch_size * num_segments * points_per_segment * 3
    operators = planar_local_operators(
        q,
        qd,
        operands.active_strain_indices,
        operands.active_strain_scales,
        operands.reference_strain.reshape(num_segments, 3),
        operands.local_points,
        jnp.asarray(
            [operands.global_eps, operands.tangent_eps], dtype=jnp.float64
        ),
        output_dims={
            "adjoint_inverse": (operator_rows, 3),
            "transported_tangent": (operator_rows, 3),
            "local_velocity": (operator_rows, 1),
            "transported_tangent_dot_velocity": (operator_rows, 1),
        },
    )
    state_rows = batch_size * 3
    outputs = planar_persistent_chain(
        *operators,
        operands.active_strain_indices,
        operands.active_strain_scales,
        operands.active_dof_ends,
        qd,
        operands.inertia_upper_rows,
        operands.inertia_upper_columns,
        operands.weighted_mass_diagonals.reshape(
            num_segments * operands.num_gauss_points, 3
        ),
        operands.gravity_base,
        operands.block_dim,
        output_dims={
            "jacobian_first": (state_rows, num_dofs),
            "derivative_first": (state_rows, 1),
            "gravity_first": (state_rows, 1),
            "jacobian_second": (state_rows, num_dofs),
            "derivative_second": (state_rows, 1),
            "gravity_second": (state_rows, 1),
            "inertia": (batch_size, num_dofs, num_dofs),
            "coriolis_qd": (batch_size, num_dofs),
            "gravity_force": (batch_size, num_dofs),
        },
    )
    return outputs[-3], outputs[-2], outputs[-1]


def execute_dynamics_terms(
    operands: PCSOperands,
    q: Array,
    qd: Array,
) -> tuple[Array, Array, Array]:
    """Dispatch to the planar or spatial constant-strain pipeline."""

    if operands.is_planar:
        return _planar_dynamics_terms(operands, q, qd)
    batch_size = q.shape[0]
    num_segments = operands.num_segments
    num_dofs = operands.num_dofs
    points_per_segment = operands.num_gauss_points + 1
    operator_rows = batch_size * num_segments * points_per_segment * 6
    operators = spatial_local_operators(
        q,
        qd,
        operands.active_strain_indices,
        operands.active_strain_scales,
        operands.reference_strain.reshape(num_segments, 6),
        operands.local_points,
        output_dims={
            "adjoint_inverse": (operator_rows, 6),
            "transported_tangent": (operator_rows, 6),
            "local_velocity": (operator_rows, 1),
            "transported_tangent_dot_velocity": (operator_rows, 1),
        },
    )
    state_rows = batch_size * 6
    outputs = spatial_persistent_chain(
        *operators,
        operands.active_strain_indices,
        operands.active_strain_scales,
        operands.active_dof_ends,
        qd,
        operands.inertia_upper_rows,
        operands.inertia_upper_columns,
        operands.weighted_mass_diagonals.reshape(
            num_segments * operands.num_gauss_points, 6
        ),
        operands.gravity_base,
        operands.block_dim,
        output_dims={
            "jacobian_first": (state_rows, num_dofs),
            "derivative_first": (state_rows, 1),
            "velocity_first": (state_rows, 1),
            "gravity_first": (state_rows, 1),
            "jacobian_second": (state_rows, num_dofs),
            "derivative_second": (state_rows, 1),
            "velocity_second": (state_rows, 1),
            "gravity_second": (state_rows, 1),
            "inertia": (batch_size, num_dofs, num_dofs),
            "coriolis_qd": (batch_size, num_dofs),
            "gravity_force": (batch_size, num_dofs),
        },
    )
    return outputs[-3], outputs[-2], outputs[-1]


__all__ = ["execute_dynamics_terms"]
