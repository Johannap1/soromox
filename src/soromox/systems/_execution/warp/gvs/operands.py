"""Explicit runtime data contract for the GVS Warp executor."""

from __future__ import annotations

from typing import Protocol

import equinox as eqx
from jax import Array


class GVSOperandSource(Protocol):
    """Precomputed GVS fields consumed by the Warp dynamics pipeline."""

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
    """Minimal shape-generic data passed to the GVS Warp executor."""

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
    def from_model(
        cls, model: GVSOperandSource, *, block_dim: int
    ) -> GVSOperands:
        """Reference precomputed model arrays without copying them."""

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


__all__ = ["GVSOperandSource", "GVSOperands"]
