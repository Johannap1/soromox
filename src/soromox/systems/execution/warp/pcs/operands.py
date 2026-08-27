"""Explicit runtime data contract for PlanarPCS and PCS Warp execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import equinox as eqx
from jax import Array


class PCSOperandSource(Protocol):
    """Structural source of precomputed PCS Warp operands.

    Both PlanarPCS and PCS satisfy this contract. Keeping the contract in the
    execution package avoids importing either concrete class and lets their
    constant-strain pipelines share orchestration without circular imports.
    """

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
    """Minimal constant-strain data passed to a PCS Warp executor.

    Static fields select planar versus spatial kernels and determine launch
    shapes. Array fields are runtime model data shared with JAX, allowing one
    executor interface to cover PlanarPCS and PCS without embedding a model
    object in the Warp layer.

    External Warp-native integrations may convert the array fields to zero-copy
    Warp views with :func:`warp.from_jax` during solver construction and retain
    them for steady-state stepping. The object itself performs no allocation or
    device transfer.

    Attributes:
        is_planar: Whether the operands describe PlanarPCS (SE(2)) rather than
            spatial PCS (SE(3)).
        num_segments: Number of serial constant-strain segments.
        num_dofs: Number of active generalized coordinates.
        num_gauss_points: Quadrature points per segment. Production Warp
            execution currently requires five.
        block_dim: Cooperative lanes per persistent chain block.
        active_strain_indices: Active-coordinate index of every full strain
            component, with negative entries for inactive components.
        active_strain_scales: Length-normalization scale of each strain row.
        reference_strain: Reference strain of every segment.
        local_points: Segment-local operator and quadrature coordinates.
        active_dof_ends: Cumulative active coordinate count after each segment.
        inertia_upper_rows: Row indices for packed upper inertia assembly.
        inertia_upper_columns: Columns matching ``inertia_upper_rows``.
        weighted_mass_diagonals: Quadrature-weighted diagonal spatial inertias.
        gravity_base: Gravity acceleration expressed in the base frame.
        global_eps: Small-angle threshold used by planar exponential operators.
        tangent_eps: Small-angle derivative threshold used by planar operators.
    """

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
        """Build an operand view over a precomputed PCS-compatible model.

        Args:
            model: Planar or spatial PCS model satisfying
                :class:`PCSOperandSource`.

        Returns:
            Operand bundle referencing the model's existing arrays. No
            numerical preprocessing or array copy is performed here.
        """

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


@dataclass(frozen=True)
class PCSPipelineShapes:
    """Describe preallocated arrays for a planar or spatial PCS Warp pipeline.

    Attributes:
        batch_size: Number of independent environments.
        num_segments: Number of serial PCS segments.
        num_dofs: Number of active generalized coordinates.
        num_gauss_points: Quadrature points per segment.
        spatial_dim: Three for PlanarPCS and six for spatial PCS.
    """

    batch_size: int
    num_segments: int
    num_dofs: int
    num_gauss_points: int
    spatial_dim: int

    @classmethod
    def from_operands(
        cls, operands: PCSOperands, *, batch_size: int
    ) -> PCSPipelineShapes:
        """Construct pipeline shapes from public runtime operands.

        Args:
            operands: Runtime dimensions and model data prepared by
                :meth:`PCSOperands.from_model`.
            batch_size: Positive number of independent environments.

        Returns:
            Immutable planar or spatial PCS allocation contract.

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
            num_gauss_points=operands.num_gauss_points,
            spatial_dim=3 if operands.is_planar else 6,
        )

    def operator_outputs(self) -> dict[str, tuple[int, ...]]:
        """Return output shapes for the dimension-specific operator launch.

        Returns:
            Mapping from local-operator output names to Warp array shapes.
        """

        rows = (
            self.batch_size
            * self.num_segments
            * (self.num_gauss_points + 1)
            * self.spatial_dim
        )
        return {
            "adjoint_inverse": (rows, self.spatial_dim),
            "transported_tangent": (rows, self.spatial_dim),
            "local_velocity": (rows, 1),
            "transported_tangent_dot_velocity": (rows, 1),
        }

    def chain_outputs(self) -> dict[str, tuple[int, ...]]:
        """Return workspace and result shapes for the persistent chain.

        Returns:
            Mapping from dimension-specific chain output names to Warp array
            shapes. Ping-pong state arrays precede final dynamics outputs.
        """

        rows = self.batch_size * self.spatial_dim
        outputs = {
            "jacobian_first": (rows, self.num_dofs),
            "derivative_first": (rows, 1),
            "gravity_first": (rows, 1),
            "jacobian_second": (rows, self.num_dofs),
            "derivative_second": (rows, 1),
            "gravity_second": (rows, 1),
        }
        if self.spatial_dim == 6:
            outputs = {
                "jacobian_first": (rows, self.num_dofs),
                "derivative_first": (rows, 1),
                "velocity_first": (rows, 1),
                "gravity_first": (rows, 1),
                "jacobian_second": (rows, self.num_dofs),
                "derivative_second": (rows, 1),
                "velocity_second": (rows, 1),
                "gravity_second": (rows, 1),
            }
        outputs.update(
            {
                "inertia": (self.batch_size, self.num_dofs, self.num_dofs),
                "coriolis_qd": (self.batch_size, self.num_dofs),
                "gravity_force": (self.batch_size, self.num_dofs),
            }
        )
        return outputs


__all__ = ["PCSOperandSource", "PCSOperands", "PCSPipelineShapes"]
