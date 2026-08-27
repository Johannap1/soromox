"""Explicit runtime data contract for PlanarPCS and PCS Warp execution."""

from __future__ import annotations

from typing import Protocol

import equinox as eqx
from jax import Array


class PCSOperandSource(Protocol):
    """Precomputed PCS fields consumed by the Warp dynamics pipelines."""

    is_planar: bool
    num_segments: int
    num_dofs: int
    num_gauss_points: int
    warp_block_dim: int
    active_strain_indices: Array
    active_strain_scales: Array
    xi_ref: Array
    dynamics_local_points: Array
    active_dof_ends: Array
    inertia_upper_rows: Array
    inertia_upper_columns: Array
    weighted_mass_diagonals: Array
    gravity_base: Array
    global_eps: float
    tangent_eps: float


class PCSOperands(eqx.Module):
    """Minimal constant-strain data passed to a PCS Warp executor."""

    is_planar: bool = eqx.field(static=True)
    num_segments: int = eqx.field(static=True)
    num_dofs: int = eqx.field(static=True)
    num_gauss_points: int = eqx.field(static=True)
    block_dim: int = eqx.field(static=True)
    active_strain_indices: Array
    active_strain_scales: Array
    reference_strain: Array
    local_points: Array
    active_dof_ends: Array
    inertia_upper_rows: Array
    inertia_upper_columns: Array
    weighted_mass_diagonals: Array
    gravity_base: Array
    global_eps: float
    tangent_eps: float

    @classmethod
    def from_model(cls, model: PCSOperandSource) -> PCSOperands:
        """Reference precomputed model arrays without copying them."""

        return cls(
            is_planar=model.is_planar,
            num_segments=model.num_segments,
            num_dofs=model.num_dofs,
            num_gauss_points=model.num_gauss_points,
            block_dim=model.warp_block_dim,
            active_strain_indices=model.active_strain_indices,
            active_strain_scales=model.active_strain_scales,
            reference_strain=model.xi_ref,
            local_points=model.dynamics_local_points,
            active_dof_ends=model.active_dof_ends,
            inertia_upper_rows=model.inertia_upper_rows,
            inertia_upper_columns=model.inertia_upper_columns,
            weighted_mass_diagonals=model.weighted_mass_diagonals,
            gravity_base=model.gravity_base,
            global_eps=model.global_eps,
            tangent_eps=model.tangent_eps,
        )


__all__ = ["PCSOperandSource", "PCSOperands"]
