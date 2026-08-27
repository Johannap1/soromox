"""Family executor for persistent batched GVS dynamics in Warp."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from soromox.systems._execution.warp.gvs.cell import scalable_cell_terms
from soromox.systems._execution.warp.gvs.chain import scalable_persistent_chain
from soromox.systems._execution.warp.gvs.joint import scalable_joint_terms
from soromox.systems._execution.warp.gvs.joint_cooperative import (
    cooperative_joint_terms,
)
from soromox.systems._execution.warp.gvs.operands import GVSOperands

SPATIAL_DIM = 6


def _gather_local(values: Array, local_to_global: Array) -> Array:
    """Gather padded segment-local coordinates from a batched active state."""

    safe_indices = jnp.maximum(local_to_global, 0)
    gathered = jnp.take(values, safe_indices, axis=1)
    return gathered * (local_to_global >= 0)[None, :, :]


def _joint_terms(
    operands: GVSOperands, q: Array, qd: Array, *, cooperative: bool
) -> tuple[Array, Array, Array, Array, Array]:
    """Evaluate all general-joint Lie terms in one runtime-shaped launch."""

    batch_size = q.shape[0]
    num_segments = operands.num_segments
    max_dof = operands.max_dof
    num_dofs = operands.num_dofs
    work_items = batch_size * num_segments
    matrix_rows = work_items * SPATIAL_DIM
    joint_terms = cooperative_joint_terms if cooperative else scalable_joint_terms
    outputs = joint_terms(
        q,
        qd,
        operands.joint_basis.reshape(num_segments * SPATIAL_DIM, max_dof),
        operands.joint_reference,
        operands.joint_local_to_global,
        output_dims={
            "adjoint": (matrix_rows, SPATIAL_DIM),
            "adjoint_dot": (matrix_rows, SPATIAL_DIM),
            "tangent_local": (matrix_rows, max_dof),
            "tangent_dot_qd": (matrix_rows, 1),
            "joint_velocity": (matrix_rows, 1),
        },
    )
    leading = (batch_size, num_segments, SPATIAL_DIM)
    tangent_local = outputs[2].reshape(*leading, max_dof)
    local_columns = jnp.maximum(operands.joint_global_to_local, 0)
    local_columns = jnp.broadcast_to(
        local_columns[None, :, None, :],
        (batch_size, num_segments, SPATIAL_DIM, num_dofs),
    )
    tangent_active = jnp.take_along_axis(tangent_local, local_columns, axis=-1)
    tangent_active *= (operands.joint_global_to_local >= 0)[None, :, None, :]
    return (
        outputs[0].reshape(*leading, SPATIAL_DIM),
        outputs[1].reshape(*leading, SPATIAL_DIM),
        tangent_active,
        outputs[3].reshape(*leading),
        outputs[4].reshape(*leading),
    )


def _cell_terms(
    operands: GVSOperands, q_link: Array, qd_link: Array
) -> tuple[Array, Array, Array, Array, Array]:
    """Evaluate spatially varying link-cell Lie terms in local coordinates."""

    batch_size = q_link.shape[0]
    num_segments = operands.num_segments
    num_cells = operands.num_cells
    max_dof = operands.max_dof
    work_items = batch_size * num_segments * num_cells
    matrix_rows = work_items * SPATIAL_DIM
    outputs = scalable_cell_terms(
        q_link.reshape(batch_size * num_segments, max_dof),
        qd_link.reshape(batch_size * num_segments, max_dof),
        operands.link_basis_z1_values.reshape(num_segments * num_cells, max_dof),
        operands.link_basis_z2_values.reshape(num_segments * num_cells, max_dof),
        operands.link_basis_rows,
        operands.link_reference_z1.reshape(num_segments * num_cells, SPATIAL_DIM),
        operands.link_reference_z2.reshape(num_segments * num_cells, SPATIAL_DIM),
        operands.segment_lengths,
        operands.cell_widths.reshape(num_segments * num_cells),
        jnp.asarray([num_cells], dtype=jnp.int32),
        jnp.asarray([0], dtype=jnp.int32),
        output_dims={
            "adjoint": (matrix_rows, SPATIAL_DIM),
            "tangent_local": (matrix_rows, max_dof),
            "link_velocity": (matrix_rows, 1),
            "step_velocity": (matrix_rows, 1),
            "tangent_velocity_dot": (matrix_rows, 1),
        },
    )
    leading = (batch_size, num_segments, num_cells, SPATIAL_DIM)
    return (
        outputs[0].reshape(*leading, SPATIAL_DIM),
        outputs[1].reshape(*leading, max_dof),
        outputs[2].reshape(*leading),
        outputs[3].reshape(*leading),
        outputs[4].reshape(*leading),
    )


def execute_dynamics_terms(
    operands: GVSOperands, q: Array, qd: Array
) -> tuple[Array, Array, Array]:
    """Assemble batched ``(B, C @ qd, G)`` with the persistent Warp pipeline."""

    (
        joint_adjoint,
        joint_adjoint_dot,
        joint_tangent,
        joint_tangent_dot_qd,
        joint_velocity,
    ) = _joint_terms(operands, q, qd, cooperative=operands.block_dim > 1)
    q_link = _gather_local(q, operands.link_local_to_global)
    qd_link = _gather_local(qd, operands.link_local_to_global)
    (
        cell_adjoint,
        cell_tangent_local,
        cell_link_velocity,
        cell_step_velocity,
        cell_tangent_velocity_dot,
    ) = _cell_terms(operands, q_link, qd_link)

    batch_size = q.shape[0]
    num_segments = operands.num_segments
    num_cells = operands.num_cells
    num_quadrature = operands.num_quadrature
    num_dofs = operands.num_dofs
    max_dof = operands.max_dof
    state_rows = batch_size * SPATIAL_DIM
    joint_rows = batch_size * num_segments * SPATIAL_DIM
    cell_rows = batch_size * num_segments * num_cells * SPATIAL_DIM
    outputs = scalable_persistent_chain(
        joint_adjoint.reshape(joint_rows, SPATIAL_DIM),
        joint_adjoint_dot.reshape(joint_rows, SPATIAL_DIM),
        joint_tangent.reshape(joint_rows, num_dofs),
        joint_tangent_dot_qd.reshape(joint_rows, 1),
        joint_velocity.reshape(joint_rows, 1),
        cell_adjoint.reshape(cell_rows, SPATIAL_DIM),
        cell_tangent_local.reshape(cell_rows, max_dof),
        cell_link_velocity.reshape(cell_rows, 1),
        cell_step_velocity.reshape(cell_rows, 1),
        cell_tangent_velocity_dot.reshape(cell_rows, 1),
        operands.link_global_to_local,
        operands.active_dofs_per_segment,
        qd,
        operands.inertia_upper_rows,
        operands.inertia_upper_columns,
        operands.weighted_mass_diagonals.reshape(
            num_segments * num_quadrature, SPATIAL_DIM
        ),
        operands.gravity_base,
        jnp.asarray([num_cells], dtype=jnp.int32),
        jnp.asarray([num_quadrature], dtype=jnp.int32),
        jnp.asarray([operands.block_dim], dtype=jnp.int32),
        output_dims={
            "jacobian_first": (state_rows, num_dofs),
            "jacobian_dot_qd_first": (state_rows, 1),
            "velocity_first": (state_rows, 1),
            "gravity_first": (state_rows, 1),
            "jacobian_second": (state_rows, num_dofs),
            "jacobian_dot_qd_second": (state_rows, 1),
            "velocity_second": (state_rows, 1),
            "gravity_second": (state_rows, 1),
            "inertia": (batch_size, num_dofs, num_dofs),
            "coriolis_qd": (batch_size, num_dofs),
            "gravity_force": (batch_size, num_dofs),
        },
    )
    return outputs[-3], outputs[-2], outputs[-1]


__all__ = ["execute_dynamics_terms"]
