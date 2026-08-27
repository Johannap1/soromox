"""JAX-facing orchestration for PlanarPCS and PCS Warp dynamics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from soromox.systems.pcs._warp.planar import (
    planar_local_operators,
    planar_persistent_chain,
)
from soromox.systems.pcs._warp.spatial import (
    spatial_local_operators,
    spatial_persistent_chain,
)

if TYPE_CHECKING:
    from soromox.systems.pcs.pcs import PCS
    from soromox.systems.pcs.planar_pcs import PlanarPCS


def _planar_dynamics_terms(
    model: PlanarPCS, q: Array, qd: Array, *, block_dim: int
) -> tuple[Array, Array, Array]:
    """Execute the staged constant-strain SE(2) pipeline."""

    batch_size = q.shape[0]
    num_segments = model.num_segments
    num_dofs = model.num_dofs
    points_per_segment = model.num_gauss_points + 1
    operator_rows = batch_size * num_segments * points_per_segment * 3
    operators = planar_local_operators(
        q,
        qd,
        model.active_strain_indices,
        model.active_strain_scales,
        model.xi_ref.reshape(num_segments, 3),
        model.dynamics_local_points,
        jnp.asarray([model.global_eps, model.tangent_eps], dtype=jnp.float64),
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
        model.active_strain_indices,
        model.active_strain_scales,
        model.active_dof_ends,
        qd,
        model.inertia_upper_rows,
        model.inertia_upper_columns,
        model.weighted_mass_diagonals.reshape(
            num_segments * model.num_gauss_points, 3
        ),
        model.gravity_base,
        block_dim,
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


def dynamics_terms(
    model: PlanarPCS | PCS,
    q: Array,
    qd: Array,
    *,
    block_dim: int,
) -> tuple[Array, Array, Array]:
    """Dispatch to the planar or spatial constant-strain pipeline."""

    if model.is_planar:
        return _planar_dynamics_terms(
            model, q, qd, block_dim=block_dim
        )
    batch_size = q.shape[0]
    num_segments = model.num_segments
    num_dofs = model.num_dofs
    points_per_segment = model.num_gauss_points + 1
    operator_rows = batch_size * num_segments * points_per_segment * 6
    operators = spatial_local_operators(
        q,
        qd,
        model.active_strain_indices,
        model.active_strain_scales,
        model.xi_ref.reshape(num_segments, 6),
        model.dynamics_local_points,
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
        model.active_strain_indices,
        model.active_strain_scales,
        model.active_dof_ends,
        qd,
        model.inertia_upper_rows,
        model.inertia_upper_columns,
        model.weighted_mass_diagonals.reshape(
            num_segments * model.num_gauss_points, 6
        ),
        model.gravity_base,
        block_dim,
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


__all__ = ["dynamics_terms"]
