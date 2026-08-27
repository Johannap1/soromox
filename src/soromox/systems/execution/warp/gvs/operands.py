"""Explicit runtime data contract for the GVS Warp executor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import equinox as eqx
from jax import Array


class GVSOperandSource(Protocol):
    """Structural source of precomputed GVS Warp operands.

    The protocol lists only data used by the executor. It deliberately avoids
    importing the concrete :class:`GVS` model and therefore keeps the execution
    package acyclic.
    """

    num_segments: int
    num_dofs: int
    max_dof: int
    max_num_integration_points: int
    B_joint: Array
    xi_ref_joint: Array
    joint_local_to_global: Array
    joint_global_to_local: Array
    link_local_to_global: Array
    link_global_to_local: Array
    active_dofs_per_segment: Array
    scaled_B_Z1_values: Array
    scaled_B_Z2_values: Array
    link_basis_rows: Array
    xi_ref_Z1: Array
    xi_ref_Z2: Array
    segment_lengths: Array
    cell_widths: Array
    inertia_upper_rows: Array
    inertia_upper_columns: Array
    inner_weighted_mass_diagonals: Array
    gravity_base: Array


class GVSOperands(eqx.Module):
    """Minimal shape-generic data passed to the GVS Warp executor.

    Static fields determine launch and output shapes and therefore participate
    in JAX/Warp compilation caching. Array fields remain runtime operands, so
    changing physical parameters does not generate new Warp source. The bundle
    contains no concrete model reference.

    External Warp-native integrations may use this object as the authoritative
    bridge from a Soromox :class:`GVS` model to the public GVS launch functions.
    Its arrays are JAX arrays owned by the model; obtain zero-copy Warp views
    with :func:`warp.from_jax` once during solver construction, then retain the
    views and caller-owned output buffers for steady-state stepping.

    Attributes:
        num_segments: Number of serial joint-link segments.
        num_dofs: Number of active generalized coordinates.
        max_dof: Padded local coordinate width shared by joints and links.
        num_cells: Integration cells per padded segment.
        num_quadrature: Interior quadrature points per padded segment.
        block_dim: Cooperative lanes used by the persistent chain launch.
        joint_basis: Flattened joint basis matrices.
        joint_reference: Reference joint strains.
        joint_local_to_global: Padded joint-local to active-coordinate map.
        joint_global_to_local: Active-coordinate to joint-local map.
        link_local_to_global: Padded link-local to active-coordinate map.
        link_global_to_local: Active-coordinate to link-local map.
        active_dofs_per_segment: Cumulative active coordinate count after each
            segment.
        link_basis_z1_values: Sparse link basis values at first Magnus nodes.
        link_basis_z2_values: Sparse link basis values at second Magnus nodes.
        link_basis_rows: Spatial row occupied by every sparse basis column.
        link_reference_z1: Reference strains at first Magnus nodes.
        link_reference_z2: Reference strains at second Magnus nodes.
        segment_lengths: Physical length of every segment.
        cell_widths: Width of every padded integration cell.
        inertia_upper_rows: Row indices for packed upper-triangular inertia
            assembly.
        inertia_upper_columns: Column indices matching ``inertia_upper_rows``.
        weighted_mass_diagonals: Quadrature-weighted diagonal spatial inertia at
            every interior point.
        gravity_base: Spatial gravity acceleration expressed in the base frame.
    """

    num_segments: int = eqx.field(static=True)
    num_dofs: int = eqx.field(static=True)
    max_dof: int = eqx.field(static=True)
    num_cells: int = eqx.field(static=True)
    num_quadrature: int = eqx.field(static=True)
    block_dim: int = eqx.field(static=True)
    joint_basis: Array
    joint_reference: Array
    joint_local_to_global: Array
    joint_global_to_local: Array
    link_local_to_global: Array
    link_global_to_local: Array
    active_dofs_per_segment: Array
    link_basis_z1_values: Array
    link_basis_z2_values: Array
    link_basis_rows: Array
    link_reference_z1: Array
    link_reference_z2: Array
    segment_lengths: Array
    cell_widths: Array
    inertia_upper_rows: Array
    inertia_upper_columns: Array
    weighted_mass_diagonals: Array
    gravity_base: Array

    @classmethod
    def from_model(cls, model: GVSOperandSource, *, block_dim: int) -> GVSOperands:
        """Build an operand view over a precomputed GVS-compatible model.

        Args:
            model: Object satisfying :class:`GVSOperandSource`.
            block_dim: Cooperative threads per persistent chain block, or one
                lane for Warp CPU execution.

        Returns:
            Operand bundle referencing the model's existing JAX arrays. No
            numerical preprocessing or array copy is performed here.
        """

        return cls(
            num_segments=model.num_segments,
            num_dofs=model.num_dofs,
            max_dof=model.max_dof,
            num_cells=model.max_num_integration_points - 1,
            num_quadrature=model.max_num_integration_points - 2,
            block_dim=block_dim,
            joint_basis=model.B_joint,
            joint_reference=model.xi_ref_joint,
            joint_local_to_global=model.joint_local_to_global,
            joint_global_to_local=model.joint_global_to_local,
            link_local_to_global=model.link_local_to_global,
            link_global_to_local=model.link_global_to_local,
            active_dofs_per_segment=model.active_dofs_per_segment,
            link_basis_z1_values=model.scaled_B_Z1_values,
            link_basis_z2_values=model.scaled_B_Z2_values,
            link_basis_rows=model.link_basis_rows,
            link_reference_z1=model.xi_ref_Z1,
            link_reference_z2=model.xi_ref_Z2,
            segment_lengths=model.segment_lengths,
            cell_widths=model.cell_widths,
            inertia_upper_rows=model.inertia_upper_rows,
            inertia_upper_columns=model.inertia_upper_columns,
            weighted_mass_diagonals=model.inner_weighted_mass_diagonals,
            gravity_base=model.gravity_base,
        )


@dataclass(frozen=True)
class GVSPipelineShapes:
    """Describe every array shape in the public GVS Warp pipeline.

    External integrators can construct this object once for their fixed batch
    size, allocate Warp arrays for each returned output mapping, and reuse those
    buffers for every step and CUDA graph replay. The names match the arguments
    of :func:`launch_joint_terms`, :func:`launch_cell_terms`, and
    :func:`launch_persistent_chain`.

    Attributes:
        batch_size: Number of independent environments.
        num_segments: Number of serial joint-link segments.
        num_dofs: Number of active generalized coordinates.
        max_dof: Padded local coordinate width.
        num_cells: Integration cells per segment.
    """

    batch_size: int
    num_segments: int
    num_dofs: int
    max_dof: int
    num_cells: int

    @classmethod
    def from_operands(
        cls, operands: GVSOperands, *, batch_size: int
    ) -> GVSPipelineShapes:
        """Construct pipeline shapes from public runtime operands.

        Args:
            operands: Runtime dimensions and model data prepared by
                :meth:`GVSOperands.from_model`.
            batch_size: Positive number of independent environments.

        Returns:
            Immutable collection of pipeline dimensions and output mappings.

        Raises:
            TypeError: If ``batch_size`` is not an integer or is a boolean.
            ValueError: If ``batch_size`` is not positive.
        """

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        return cls(
            batch_size=batch_size,
            num_segments=operands.num_segments,
            num_dofs=operands.num_dofs,
            max_dof=operands.max_dof,
            num_cells=operands.num_cells,
        )

    @property
    def local_state(self) -> tuple[int, int]:
        """Shape of padded ``q_link`` and ``qd_link`` workspace arrays."""

        return self.batch_size * self.num_segments, self.max_dof

    def joint_outputs(self) -> dict[str, tuple[int, ...]]:
        """Return output shapes for a joint-terms launch.

        Returns:
            Mapping from public joint output argument names to array shapes.
        """

        rows = self.batch_size * self.num_segments * 6
        return {
            "adjoint": (rows, 6),
            "adjoint_dot": (rows, 6),
            "tangent_local": (rows, self.max_dof),
            "tangent_dot_qd": (rows, 1),
            "joint_velocity": (rows, 1),
        }

    def cell_outputs(self) -> dict[str, tuple[int, ...]]:
        """Return output shapes for a link-cell terms launch.

        Returns:
            Mapping from public cell output argument names to array shapes.
        """

        rows = self.batch_size * self.num_segments * self.num_cells * 6
        return {
            "adjoint": (rows, 6),
            "tangent_local": (rows, self.max_dof),
            "link_velocity": (rows, 1),
            "step_velocity": (rows, 1),
            "tangent_velocity_dot": (rows, 1),
        }

    def chain_outputs(self) -> dict[str, tuple[int, ...]]:
        """Return scratch and final output shapes for a persistent-chain launch.

        Returns:
            Mapping from public persistent-chain output argument names to array
            shapes. The first eight arrays are caller-owned ping-pong workspace;
            the last three are the public dynamics results.
        """

        rows = self.batch_size * 6
        return {
            "jacobian_first": (rows, self.num_dofs),
            "jacobian_dot_qd_first": (rows, 1),
            "velocity_first": (rows, 1),
            "gravity_first": (rows, 1),
            "jacobian_second": (rows, self.num_dofs),
            "jacobian_dot_qd_second": (rows, 1),
            "velocity_second": (rows, 1),
            "gravity_second": (rows, 1),
            "inertia": (self.batch_size, self.num_dofs, self.num_dofs),
            "coriolis_qd": (self.batch_size, self.num_dofs),
            "gravity_force": (self.batch_size, self.num_dofs),
        }


__all__ = ["GVSOperandSource", "GVSOperands", "GVSPipelineShapes"]
