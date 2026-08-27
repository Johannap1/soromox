"""Family executor for persistent batched GVS dynamics in Warp."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from soromox.systems.execution.warp.gvs.cell import scalable_cell_terms
from soromox.systems.execution.warp.gvs.chain import scalable_persistent_chain
from soromox.systems.execution.warp.gvs.joint import scalable_joint_terms
from soromox.systems.execution.warp.gvs.joint_cooperative import (
    cooperative_joint_terms,
)
from soromox.systems.execution.warp.gvs.operands import (
    GVSOperands,
    GVSPipelineShapes,
)

SPATIAL_DIM = 6


def _gather_local(values: Array, local_to_global: Array) -> Array:
    """Gather padded segment-local coordinates from batched active states.

    Args:
        values: Active coordinates with shape ``(batch_size, num_dofs)``.
        local_to_global: Per-segment map from padded local columns to active
            global columns. Negative entries denote padding.

    Returns:
        Padded local values with shape
        ``(batch_size, num_segments, max_dof)``. Padding entries are zero.
    """

    safe_indices = jnp.maximum(local_to_global, 0)
    gathered = jnp.take(values, safe_indices, axis=1)
    return gathered * (local_to_global >= 0)[None, :, :]


def _joint_terms(
    operands: GVSOperands, q: Array, qd: Array, *, cooperative: bool
) -> tuple[Array, Array, Array, Array, Array]:
    """Evaluate all general-joint Lie terms in one runtime-shaped launch.

    Args:
        operands: Static dimensions, joint bases, references, and coordinate
            maps prepared from the model.
        q: Batched active generalized coordinates.
        qd: Batched active generalized velocities.
        cooperative: Whether each joint work item uses a cooperative CUDA block
            rather than the one-lane CPU-compatible kernel.

    Returns:
        Joint adjoints, adjoint derivatives, active-coordinate tangents,
        tangent-derivative actions, and joint velocities, each reshaped by
        environment and segment.
    """

    batch_size = q.shape[0]
    num_segments = operands.num_segments
    max_dof = operands.max_dof
    num_dofs = operands.num_dofs
    output_dims = GVSPipelineShapes.from_operands(
        operands, batch_size=batch_size
    ).joint_outputs()
    joint_terms = cooperative_joint_terms if cooperative else scalable_joint_terms
    outputs = joint_terms(
        q,
        qd,
        operands.joint_basis.reshape(num_segments * SPATIAL_DIM, max_dof),
        operands.joint_reference,
        operands.joint_local_to_global,
        output_dims=output_dims,
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
    """Evaluate spatially varying link-cell Lie terms in local coordinates.

    Args:
        operands: GVS runtime operand bundle.
        q_link: Batched padded link coordinates with shape
            ``(batch_size, num_segments, max_dof)``.
        qd_link: Link velocities with the same shape as ``q_link``.

    Returns:
        Cell adjoints, local tangents, link velocities, step velocities, and
        tangent-velocity derivatives for every environment, segment, and cell.
    """

    batch_size = q_link.shape[0]
    num_segments = operands.num_segments
    num_cells = operands.num_cells
    max_dof = operands.max_dof
    output_dims = GVSPipelineShapes.from_operands(
        operands, batch_size=batch_size
    ).cell_outputs()
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
        output_dims=output_dims,
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
    """Assemble batched GVS dynamics with the persistent Warp pipeline.

    General-joint and spatially varying cell operators are evaluated in
    parallel first. One persistent cooperative block per environment then walks
    the serial segment chain, retains recurrence state on chip, accumulates the
    five-point quadrature contributions, and writes only the final inertia,
    convective-force, and gravity-force terms.

    Args:
        operands: Shape-generic GVS runtime operand bundle.
        q: FP64 generalized coordinates with shape
            ``(batch_size, num_dofs)``.
        qd: FP64 generalized velocities with the same shape as ``q``.

    Returns:
        Tuple ``(B, Cqd, G)`` with shapes
        ``(batch_size, num_dofs, num_dofs)``,
        ``(batch_size, num_dofs)``, and ``(batch_size, num_dofs)``.
    """

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
    joint_rows = batch_size * num_segments * SPATIAL_DIM
    cell_rows = batch_size * num_segments * num_cells * SPATIAL_DIM
    output_dims = GVSPipelineShapes.from_operands(
        operands, batch_size=batch_size
    ).chain_outputs()
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
        output_dims=output_dims,
    )
    return outputs[-3], outputs[-2], outputs[-1]


__all__ = ["execute_dynamics_terms"]
