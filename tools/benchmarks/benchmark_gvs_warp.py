#!/usr/bin/env python3
"""Correctness and runtime harness for experimental GVS Warp kernels."""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root / "src"))
    sys.path.insert(1, str(repository_root))

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

jax.config.update("jax_enable_x64", True)

from soromox.systems import GVS, JointSpec
from soromox.utils.lie_algebra import se3
from tools.benchmarks._benchmark_common import (
    _gvs_context,
    _gvs_factory,
    _gvs_segment,
)
from tools.benchmarks.gvs_warp_assembly_kernels import (
    scalable_assemble_dynamics,
)
from tools.benchmarks.gvs_warp_joint_kernels import scalable_joint_terms
from tools.benchmarks.gvs_warp_kernels import (
    MAX_DOF,
    NUM_CELLS,
    NUM_DOFS,
    NUM_QUADRATURE_CELLS,
    NUM_SEGMENTS,
    SPATIAL_DIM,
    contract_cell_terms,
    cooperative_segment_dynamics,
    fused_cell_advance,
    persistent_raw_dynamics,
    segment_recurrence,
)
from tools.benchmarks.gvs_warp_lie_kernels import (
    matrix_free_cell_terms,
    matrix_free_cell_terms_serial,
)
from tools.benchmarks.gvs_warp_persistent_kernels import (
    scalable_persistent_chain,
)
from tools.benchmarks.gvs_warp_scalable_kernels import (
    scalable_cell_terms,
    scalable_segment_recurrence,
)
from tools.benchmarks.gvs_warp_segment_kernels import (
    scalable_cooperative_segment,
)
from tools.benchmarks.gvs_warp_solve_kernels import scalable_cholesky_solve
from tools.benchmarks.gvs_warp_specialized_kernels import (
    constant_strain_cell_terms,
)

Array = jax.Array
Tree = Any
JOINT_TYPES = (
    "fixed",
    "revolute",
    "prismatic",
    "helical",
    "cylindrical",
    "planar",
    "spherical",
    "free",
    "mixed",
)


@dataclass(frozen=True)
class ScalableWarpModelData:
    """Persistent shape-generic arrays captured by the experimental kernels."""

    joint_basis: Array
    joint_reference: Array
    joint_local_to_global: Array
    joint_global_to_local: Array
    link_local_to_global: Array
    basis_z1: Array
    basis_z2: Array
    reference_z1: Array
    reference_z2: Array
    segment_lengths: Array
    cell_widths: Array


def _build_scalable_model_data(robot: Any) -> ScalableWarpModelData:
    """Flatten static model data without specializing generated Warp source."""

    num_segments = int(robot.B_joint.shape[0])
    num_dofs = int(robot.num_dofs)
    max_dof = int(robot.B_joint.shape[-1])
    gather_indices = np.asarray(robot.gather_indices)
    gather_mask = np.asarray(robot.gather_mask)

    local_to_global = np.where(gather_mask, gather_indices, -1).astype(np.int32)
    joint_global_to_local = np.full(
        (num_segments, num_dofs), -1, dtype=np.int32
    )
    for segment in range(num_segments):
        for local_column in range(max_dof):
            global_column = local_to_global[segment, 0, local_column]
            if global_column >= 0:
                joint_global_to_local[segment, global_column] = local_column

    basis_z1 = robot.B_Z1
    basis_z2 = robot.B_Z2
    if robot.scale_rotational_basis_by_length:
        scales = robot.segment_lengths[:, None, None]
        basis_z1 = basis_z1.at[:, :, :3].divide(scales)
        basis_z2 = basis_z2.at[:, :, :3].divide(scales)

    return ScalableWarpModelData(
        joint_basis=robot.B_joint,
        joint_reference=robot.xi_ref_joint,
        joint_local_to_global=jnp.asarray(local_to_global[:, 0]),
        joint_global_to_local=jnp.asarray(joint_global_to_local),
        link_local_to_global=jnp.asarray(local_to_global[:, 1]),
        basis_z1=basis_z1,
        basis_z2=basis_z2,
        reference_z1=robot.xi_ref_Z1,
        reference_z2=robot.xi_ref_Z2,
        segment_lengths=robot.segment_lengths,
        cell_widths=(
            robot.integration_points[:, 1:] - robot.integration_points[:, :-1]
        ),
    )


def _joint_spec(joint_type: str) -> JointSpec:
    """Construct a nondegenerate representative of one supported joint."""

    if joint_type == "helical":
        return JointSpec(type="helical", axis="z", pitch=0.05)
    if joint_type == "planar":
        return JointSpec(type="planar", plane="xy")
    if joint_type in {"revolute", "prismatic", "cylindrical"}:
        return JointSpec(type=joint_type, axis="z")
    return JointSpec(type=joint_type)


def _block(tree: Tree) -> None:
    jax.tree.map(lambda value: value.block_until_ready(), tree)


def _coadjoint_action(velocity: Array, momentum: Array) -> Array:
    omega = velocity[..., :3]
    linear_velocity = velocity[..., 3:]
    moment = momentum[..., :3]
    force = momentum[..., 3:]
    return jnp.concatenate(
        [
            jnp.cross(omega, moment) + jnp.cross(linear_velocity, force),
            jnp.cross(omega, force),
        ],
        axis=-1,
    )


def _build_precompute(robot: Any) -> Callable[[Array, Array], tuple[Array, ...]]:
    segment_indices = jnp.arange(NUM_SEGMENTS)
    cell_indices = jnp.arange(NUM_CELLS)

    def precompute_one(q: Array, qd: Array) -> tuple[Array, ...]:
        q_blocks = robot._min_size_gathered(q)
        qd_blocks = robot._min_size_gathered(qd)

        def insert_basis(local_basis: Array, segment: Array, block: int) -> Array:
            indices = jnp.minimum(robot.gather_indices[segment, block], NUM_DOFS - 1)
            masked = local_basis * robot.gather_mask[segment, block][None, :]
            return (
                jnp.zeros((SPATIAL_DIM, NUM_DOFS), dtype=q.dtype)
                .at[:, indices]
                .add(masked)
            )

        def evaluate_joint(segment: Array) -> tuple[Array, ...]:
            (
                _,
                adjoint,
                adjoint_dot,
                tangent_basis,
                tangent_basis_dot,
                velocity,
            ) = robot._joint_jacobian_time_derivative_step_terms(
                robot.B_joint[segment],
                robot.xi_ref_joint[segment],
                q_blocks[segment, 0],
                qd_blocks[segment, 0],
            )
            tangent_active = insert_basis(tangent_basis, segment, 0)
            tangent_dot_qd = tangent_basis_dot @ qd_blocks[segment, 0]
            return (
                adjoint,
                adjoint_dot,
                tangent_active,
                tangent_dot_qd,
                velocity,
            )

        def evaluate_segment(segment: Array) -> tuple[Array, ...]:
            def evaluate_cell(cell: Array) -> tuple[Array, ...]:
                width = (
                    robot.integration_points[segment, cell + 1]
                    - robot.integration_points[segment, cell]
                )
                (
                    _,
                    tangent,
                    tangent_dot,
                    adjoint,
                    magnus_basis,
                    magnus_basis_dot,
                ) = robot._magnus_jacobian_time_derivative_step_terms(
                    robot.segment_lengths[segment],
                    width,
                    q_blocks[segment, 1],
                    qd_blocks[segment, 1],
                    robot.B_Z1[segment, cell],
                    robot.B_Z2[segment, cell],
                    robot.xi_ref_Z1[segment, cell],
                    robot.xi_ref_Z2[segment, cell],
                )
                local_tangent = tangent @ magnus_basis
                tangent_active = insert_basis(local_tangent, segment, 1)
                link_velocity = local_tangent @ qd_blocks[segment, 1]
                step_velocity = adjoint @ link_velocity
                tangent_velocity_dot = (
                    tangent_dot @ magnus_basis + tangent @ magnus_basis_dot
                ) @ qd_blocks[segment, 1]
                return (
                    adjoint,
                    tangent_active,
                    link_velocity,
                    step_velocity,
                    tangent_velocity_dot,
                    tangent,
                    tangent_dot,
                    magnus_basis,
                    magnus_basis_dot,
                )

            return jax.vmap(evaluate_cell)(cell_indices)

        return (
            *jax.vmap(evaluate_joint)(segment_indices),
            *jax.vmap(evaluate_segment)(segment_indices),
            qd_blocks[:, 1],
        )

    return jax.vmap(precompute_one)


def _build_joint_precompute(robot: Any) -> Callable[[Array, Array], tuple[Array, ...]]:
    """Evaluate only joint terms and gather link coordinates and velocities."""

    num_segments = int(robot.B_joint.shape[0])
    num_dofs = int(robot.num_dofs)
    segment_indices = jnp.arange(num_segments)

    def precompute_one(q: Array, qd: Array) -> tuple[Array, ...]:
        q_blocks = robot._min_size_gathered(q)
        qd_blocks = robot._min_size_gathered(qd)

        def insert_basis(local_basis: Array, segment: Array) -> Array:
            indices = jnp.minimum(robot.gather_indices[segment, 0], num_dofs - 1)
            masked = local_basis * robot.gather_mask[segment, 0][None, :]
            return (
                jnp.zeros((SPATIAL_DIM, num_dofs), dtype=q.dtype)
                .at[:, indices]
                .add(masked)
            )

        def evaluate_joint(segment: Array) -> tuple[Array, ...]:
            (
                _,
                adjoint,
                adjoint_dot,
                tangent_basis,
                tangent_basis_dot,
                velocity,
            ) = robot._joint_jacobian_time_derivative_step_terms(
                robot.B_joint[segment],
                robot.xi_ref_joint[segment],
                q_blocks[segment, 0],
                qd_blocks[segment, 0],
            )
            return (
                adjoint,
                adjoint_dot,
                insert_basis(tangent_basis, segment),
                tangent_basis_dot @ qd_blocks[segment, 0],
                velocity,
            )

        return (*jax.vmap(evaluate_joint)(segment_indices), q_blocks[:, 1], qd_blocks[:, 1])

    return jax.vmap(precompute_one)


def _build_fixed_joint_precompute(
    robot: Any,
) -> Callable[[Array, Array], tuple[Array, ...]]:
    """Return exact identity/zero joint terms for this fixed-joint benchmark."""

    num_segments = int(robot.B_joint.shape[0])
    num_dofs = int(robot.num_dofs)
    if (
        np.any(np.asarray(robot.B_joint))
        or np.any(np.asarray(robot.xi_ref_joint))
        or np.any(np.asarray(robot.gather_mask[:, 0]))
    ):
        raise ValueError(
            "Fixed-joint specialization requires zero joint bases, reference "
            "strains, and active joint gather masks."
        )

    def precompute(q: Array, qd: Array) -> tuple[Array, ...]:
        batch_size = q.shape[0]
        q_blocks = jax.vmap(robot._min_size_gathered)(q)
        qd_blocks = jax.vmap(robot._min_size_gathered)(qd)
        identity = jnp.broadcast_to(
            jnp.eye(SPATIAL_DIM, dtype=q.dtype),
            (batch_size, num_segments, SPATIAL_DIM, SPATIAL_DIM),
        )
        matrix_zeros = jnp.zeros_like(identity)
        tangent_zeros = jnp.zeros(
            (batch_size, num_segments, SPATIAL_DIM, num_dofs), dtype=q.dtype
        )
        vector_zeros = jnp.zeros(
            (batch_size, num_segments, SPATIAL_DIM), dtype=q.dtype
        )
        return (
            identity,
            matrix_zeros,
            tangent_zeros,
            vector_zeros,
            vector_zeros,
            q_blocks[:, :, 1],
            qd_blocks[:, :, 1],
        )

    return precompute


def _warp_scalable_joint_terms(
    q: Array,
    qd: Array,
    model_data: ScalableWarpModelData,
) -> tuple[Array, Array, Array, Array, Array]:
    """Evaluate general GVS joint terms in one shape-generic Warp launch."""

    batch_size = q.shape[0]
    num_segments, max_dof = model_data.joint_local_to_global.shape
    num_dofs = q.shape[1]
    work_items = batch_size * num_segments
    matrix_rows = work_items * SPATIAL_DIM
    outputs = scalable_joint_terms(
        q,
        qd,
        model_data.joint_basis.reshape(num_segments * SPATIAL_DIM, max_dof),
        model_data.joint_reference,
        model_data.joint_local_to_global,
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
    local_columns = jnp.maximum(model_data.joint_global_to_local, 0)
    local_columns = jnp.broadcast_to(
        local_columns[None, :, None, :],
        (batch_size, num_segments, SPATIAL_DIM, num_dofs),
    )
    tangent_active = jnp.take_along_axis(
        tangent_local, local_columns, axis=-1
    )
    tangent_active *= (
        model_data.joint_global_to_local >= 0
    )[None, :, None, :]
    return (
        outputs[0].reshape(*leading, SPATIAL_DIM),
        outputs[1].reshape(*leading, SPATIAL_DIM),
        tangent_active,
        outputs[3].reshape(*leading),
        outputs[4].reshape(*leading),
    )


def _gather_link_coordinates(
    values: Array, model_data: ScalableWarpModelData
) -> Array:
    """Gather padded link coordinates from one batched global state array."""

    local_to_global = model_data.link_local_to_global
    gathered = jnp.take(values, jnp.maximum(local_to_global, 0), axis=1)
    return gathered * (local_to_global >= 0)[None, :, :]


def _build_warp_joint_precompute(
    model_data: ScalableWarpModelData,
) -> Callable[[Array, Array], tuple[Array, ...]]:
    """Return Warp joint terms and directly gathered link coordinates."""

    def precompute(q: Array, qd: Array) -> tuple[Array, ...]:
        return (
            *_warp_scalable_joint_terms(q, qd, model_data),
            _gather_link_coordinates(q, model_data),
            _gather_link_coordinates(qd, model_data),
        )

    return precompute


def _warp_matrix_free_cell_terms(
    q_link: Array,
    qd_link: Array,
    basis_z1: Array,
    basis_z2: Array,
    reference_z1: Array,
    reference_z2: Array,
    segment_lengths: Array,
    cell_widths: Array,
    gather_indices: Array,
    gather_mask: Array,
    *,
    serial_cell: bool = False,
    constant_strain: bool = False,
) -> tuple[Array, ...]:
    batch_size = q_link.shape[0]
    work_items = batch_size * NUM_SEGMENTS * NUM_CELLS
    matrix_rows = work_items * SPATIAL_DIM
    output_dims = {
        "adjoint": (matrix_rows, SPATIAL_DIM),
        "tangent_active": (matrix_rows, NUM_DOFS),
        "link_velocity": (matrix_rows, 1),
        "step_velocity": (matrix_rows, 1),
        "tangent_velocity_dot": (matrix_rows, 1),
    }
    common_inputs = (
        q_link.reshape(batch_size * NUM_SEGMENTS, MAX_DOF),
        qd_link.reshape(batch_size * NUM_SEGMENTS, MAX_DOF),
    )
    if constant_strain:
        outputs = constant_strain_cell_terms(
            *common_inputs,
            basis_z1.reshape(NUM_SEGMENTS * NUM_CELLS * SPATIAL_DIM, MAX_DOF),
            reference_z1.reshape(NUM_SEGMENTS * NUM_CELLS, SPATIAL_DIM),
            segment_lengths,
            cell_widths.reshape(NUM_SEGMENTS * NUM_CELLS),
            output_dims=output_dims,
        )
    else:
        callable_kernel = (
            matrix_free_cell_terms_serial if serial_cell else matrix_free_cell_terms
        )
        outputs = callable_kernel(
            *common_inputs,
            basis_z1.reshape(NUM_SEGMENTS * NUM_CELLS * SPATIAL_DIM, MAX_DOF),
            basis_z2.reshape(NUM_SEGMENTS * NUM_CELLS * SPATIAL_DIM, MAX_DOF),
            reference_z1.reshape(NUM_SEGMENTS * NUM_CELLS, SPATIAL_DIM),
            reference_z2.reshape(NUM_SEGMENTS * NUM_CELLS, SPATIAL_DIM),
            segment_lengths,
            cell_widths.reshape(NUM_SEGMENTS * NUM_CELLS),
            gather_indices,
            gather_mask,
            output_dims=output_dims,
        )
    leading = (batch_size, NUM_SEGMENTS, NUM_CELLS, SPATIAL_DIM)
    return (
        outputs[0].reshape(*leading, SPATIAL_DIM),
        outputs[1].reshape(*leading, NUM_DOFS),
        outputs[2].reshape(*leading),
        outputs[3].reshape(*leading),
        outputs[4].reshape(*leading),
    )


def _build_matrix_free_precompute(
    robot: Any,
    *,
    serial_cell: bool = False,
    constant_strain: bool = False,
    fixed_joints: bool = False,
) -> Callable[[Array, Array], tuple[Array, ...]]:
    """Replace JAX cell-local Lie preparation with one matrix-free Warp kernel."""

    if constant_strain:
        expected_indices = np.arange(NUM_DOFS, dtype=np.int32).reshape(
            NUM_SEGMENTS, MAX_DOF
        )
        if (
            robot.scale_rotational_basis_by_length
            or not np.array_equal(np.asarray(robot.B_Z1), np.asarray(robot.B_Z2))
            or not np.array_equal(
                np.asarray(robot.xi_ref_Z1), np.asarray(robot.xi_ref_Z2)
            )
            or not np.array_equal(
                np.asarray(robot.gather_indices[:, 1]), expected_indices
            )
            or not np.all(np.asarray(robot.gather_mask[:, 1]))
        ):
            raise ValueError(
                "Constant-strain specialization requires identical Z1/Z2 data, "
                "unscaled rotational bases, and six contiguous active DoFs per segment."
            )

    joint_precompute = (
        _build_fixed_joint_precompute(robot)
        if fixed_joints
        else _build_joint_precompute(robot)
    )
    basis_z1 = robot.B_Z1
    basis_z2 = robot.B_Z2
    if robot.scale_rotational_basis_by_length:
        scales = robot.segment_lengths[:, None, None]
        basis_z1 = basis_z1.at[:, :, :3].divide(scales)
        basis_z2 = basis_z2.at[:, :, :3].divide(scales)
    cell_widths = robot.integration_points[:, 1:] - robot.integration_points[:, :-1]
    gather_indices = jnp.asarray(robot.gather_indices[:, 1], dtype=jnp.int32)
    gather_mask = jnp.asarray(robot.gather_mask[:, 1], dtype=robot.B_Z1.dtype)

    def precompute(q: Array, qd: Array) -> tuple[Array, ...]:
        (
            joint_adjoint,
            joint_adjoint_dot,
            joint_tangent,
            joint_tangent_dot_qd,
            joint_velocity,
            q_link,
            qd_link,
        ) = joint_precompute(q, qd)
        (
            cell_adjoint,
            cell_tangent,
            cell_link_velocity,
            cell_step_velocity,
            cell_tangent_velocity_dot,
        ) = _warp_matrix_free_cell_terms(
            q_link,
            qd_link,
            basis_z1,
            basis_z2,
            robot.xi_ref_Z1,
            robot.xi_ref_Z2,
            robot.segment_lengths,
            cell_widths,
            gather_indices,
            gather_mask,
            serial_cell=serial_cell,
            constant_strain=constant_strain,
        )
        # The four raw arrays are unused by the matrix-free Option 6 path. They
        # retain compatible positions in the shared precompute interface.
        return (
            joint_adjoint,
            joint_adjoint_dot,
            joint_tangent,
            joint_tangent_dot_qd,
            joint_velocity,
            cell_adjoint,
            cell_tangent,
            cell_link_velocity,
            cell_step_velocity,
            cell_tangent_velocity_dot,
            cell_adjoint,
            cell_adjoint,
            cell_adjoint,
            cell_adjoint,
            qd_link,
        )

    return precompute


def _warp_scalable_cell_terms(
    q_link: Array,
    qd_link: Array,
    basis_z1: Array,
    basis_z2: Array,
    reference_z1: Array,
    reference_z2: Array,
    segment_lengths: Array,
    cell_widths: Array,
    *,
    order_zero: bool,
) -> tuple[Array, ...]:
    """Call the compact local-coordinate Lie kernel."""

    batch_size, num_segments, max_dof = q_link.shape
    num_cells = reference_z1.shape[1]
    work_items = batch_size * num_segments * num_cells
    matrix_rows = work_items * SPATIAL_DIM
    outputs = scalable_cell_terms(
        q_link.reshape(batch_size * num_segments, max_dof),
        qd_link.reshape(batch_size * num_segments, max_dof),
        basis_z1.reshape(num_segments * num_cells * SPATIAL_DIM, max_dof),
        basis_z2.reshape(num_segments * num_cells * SPATIAL_DIM, max_dof),
        reference_z1.reshape(num_segments * num_cells, SPATIAL_DIM),
        reference_z2.reshape(num_segments * num_cells, SPATIAL_DIM),
        segment_lengths,
        cell_widths.reshape(num_segments * num_cells),
        jnp.asarray([num_cells], dtype=jnp.int32),
        jnp.asarray([int(order_zero)], dtype=jnp.int32),
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


def _build_scalable_precompute(
    robot: Any,
    *,
    order_zero: bool,
    fixed_joints: bool,
    warp_joints: bool = False,
) -> Callable[[Array, Array], tuple[Array, ...]]:
    """Prepare general joints and compact cell-local Lie terms."""

    if order_zero and (
        robot.scale_rotational_basis_by_length
        or not np.array_equal(np.asarray(robot.B_Z1), np.asarray(robot.B_Z2))
        or not np.array_equal(
            np.asarray(robot.xi_ref_Z1), np.asarray(robot.xi_ref_Z2)
        )
    ):
        raise ValueError(
            "Order-zero specialization requires identical Z1/Z2 bases and "
            "reference strains without rotational basis rescaling."
        )
    model_data = _build_scalable_model_data(robot)
    if fixed_joints and warp_joints:
        raise ValueError("Fixed-joint and general Warp-joint paths are exclusive")
    joint_precompute = (
        _build_fixed_joint_precompute(robot)
        if fixed_joints
        else (
            _build_warp_joint_precompute(model_data)
            if warp_joints
            else _build_joint_precompute(robot)
        )
    )

    def precompute(q: Array, qd: Array) -> tuple[Array, ...]:
        (
            joint_adjoint,
            joint_adjoint_dot,
            joint_tangent,
            joint_tangent_dot_qd,
            joint_velocity,
            q_link,
            qd_link,
        ) = joint_precompute(q, qd)
        cell_terms = _warp_scalable_cell_terms(
            q_link,
            qd_link,
            model_data.basis_z1,
            model_data.basis_z2,
            model_data.reference_z1,
            model_data.reference_z2,
            model_data.segment_lengths,
            model_data.cell_widths,
            order_zero=order_zero,
        )
        return (
            joint_adjoint,
            joint_adjoint_dot,
            joint_tangent,
            joint_tangent_dot_qd,
            joint_velocity,
            *cell_terms,
        )

    return precompute


def _warp_cell(
    adjoint: Array,
    tangent_active: Array,
    link_velocity: Array,
    step_velocity: Array,
    tangent_velocity_dot: Array,
    qd: Array,
    jacobian: Array,
    jacobian_dot_qd: Array,
    velocity: Array,
    gravity: Array,
) -> tuple[Array, Array, Array, Array]:
    return fused_cell_advance(
        adjoint,
        tangent_active,
        link_velocity,
        step_velocity,
        tangent_velocity_dot,
        qd,
        jacobian,
        jacobian_dot_qd,
        velocity,
        gravity,
        output_dims={
            "jacobian_next": jacobian.shape,
            "jacobian_dot_qd_next": jacobian_dot_qd.shape,
            "velocity_next": velocity.shape,
            "gravity_next": gravity.shape,
        },
    )


def _build_option_1(robot: Any) -> Callable[[Array, Array], tuple[Array, ...]]:
    precompute = _build_precompute(robot)
    mass_diagonals = jnp.diagonal(robot.inner_mass_matrices, axis1=-2, axis2=-1)
    gravity_initial = se3.adjoint_inverse(robot.g0) @ robot.g

    def dynamics(q: Array, qd: Array) -> tuple[Array, Array, Array]:
        (
            joint_adjoint,
            joint_adjoint_dot,
            joint_tangent,
            joint_tangent_dot_qd,
            joint_velocity,
            cell_adjoint,
            cell_tangent,
            cell_link_velocity,
            cell_step_velocity,
            cell_tangent_velocity_dot,
            _cell_tangent_raw,
            _cell_tangent_dot_raw,
            _cell_magnus_basis_raw,
            _cell_magnus_basis_dot_raw,
            _qd_link,
        ) = precompute(q, qd)
        batch_size = q.shape[0]
        jacobian = jnp.zeros((batch_size, SPATIAL_DIM, NUM_DOFS), dtype=q.dtype)
        vector = jnp.zeros((batch_size, SPATIAL_DIM), dtype=q.dtype)
        jacobian_dot_qd = vector
        velocity = vector
        gravity = jnp.broadcast_to(gravity_initial, (batch_size, SPATIAL_DIM))
        inertia = jnp.zeros((batch_size, NUM_DOFS, NUM_DOFS), dtype=q.dtype)
        coriolis_qd = jnp.zeros((batch_size, NUM_DOFS), dtype=q.dtype)
        gravity_force = jnp.zeros((batch_size, NUM_DOFS), dtype=q.dtype)

        for segment in range(NUM_SEGMENTS):
            adjoint = joint_adjoint[:, segment]
            jacobian_tip_qd = jnp.einsum("bij,bj->bi", jacobian, qd)
            jacobian = jnp.einsum(
                "bij,bjk->bik", adjoint, jacobian + joint_tangent[:, segment]
            )
            jacobian_dot_qd = jnp.einsum(
                "bij,bj->bi",
                adjoint,
                jacobian_dot_qd + joint_tangent_dot_qd[:, segment],
            ) + jnp.einsum("bij,bj->bi", joint_adjoint_dot[:, segment], jacobian_tip_qd)
            velocity = jnp.einsum(
                "bij,bj->bi", adjoint, velocity + joint_velocity[:, segment]
            )
            gravity = jnp.einsum("bij,bj->bi", adjoint, gravity)

            quadrature_states: list[tuple[Array, Array, Array, Array]] = []
            for cell in range(NUM_CELLS):
                jacobian, jacobian_dot_qd, velocity, gravity = _warp_cell(
                    cell_adjoint[:, segment, cell],
                    cell_tangent[:, segment, cell],
                    cell_link_velocity[:, segment, cell],
                    cell_step_velocity[:, segment, cell],
                    cell_tangent_velocity_dot[:, segment, cell],
                    qd,
                    jacobian,
                    jacobian_dot_qd,
                    velocity,
                    gravity,
                )
                if cell < NUM_QUADRATURE_CELLS:
                    quadrature_states.append(
                        (jacobian, jacobian_dot_qd, velocity, gravity)
                    )

            jacobians = jnp.stack([state[0] for state in quadrature_states], axis=1)
            jacobians_dot_qd = jnp.stack(
                [state[1] for state in quadrature_states], axis=1
            )
            velocities = jnp.stack([state[2] for state in quadrature_states], axis=1)
            gravities = jnp.stack([state[3] for state in quadrature_states], axis=1)
            weights = robot.inner_integration_weights[segment]
            masses = mass_diagonals[segment]
            weighted_masses = weights[:, None] * masses
            inertia += jnp.einsum(
                "bqri,qr,bqrj->bij", jacobians, weighted_masses, jacobians
            )
            momentum = masses[None, :, :] * velocities
            wrench = masses[None, :, :] * jacobians_dot_qd + _coadjoint_action(
                velocities, momentum
            )
            coriolis_qd += jnp.einsum(
                "bqri,bqr->bi", jacobians, weights[None, :, None] * wrench
            )
            gravity_force -= jnp.einsum(
                "bqri,bqr->bi",
                jacobians,
                weighted_masses[None, :, :] * gravities,
            )

        return inertia, coriolis_qd, gravity_force

    return dynamics


def _warp_segment(
    adjoint: Array,
    tangent_active: Array,
    link_velocity: Array,
    step_velocity: Array,
    tangent_velocity_dot: Array,
    qd: Array,
    jacobian: Array,
    jacobian_dot_qd: Array,
    velocity: Array,
    gravity: Array,
) -> tuple[Array, Array, Array, Array]:
    batch_size = jacobian.shape[0]
    return segment_recurrence(
        adjoint,
        tangent_active,
        link_velocity,
        step_velocity,
        tangent_velocity_dot,
        qd,
        jacobian,
        jacobian_dot_qd,
        velocity,
        gravity,
        output_dims={
            "jacobian_states": (
                batch_size,
                NUM_CELLS,
                SPATIAL_DIM,
                NUM_DOFS,
            ),
            "jacobian_dot_qd_states": (
                batch_size,
                NUM_CELLS,
                SPATIAL_DIM,
            ),
            "velocity_states": (batch_size, NUM_CELLS, SPATIAL_DIM),
            "gravity_states": (batch_size, NUM_CELLS, SPATIAL_DIM),
        },
    )


def _warp_scalable_segment(
    adjoint: Array,
    tangent_local: Array,
    link_velocity: Array,
    step_velocity: Array,
    tangent_velocity_dot: Array,
    global_to_local: Array,
    qd: Array,
    jacobian: Array,
    jacobian_dot_qd: Array,
    velocity: Array,
    gravity: Array,
) -> tuple[Array, Array, Array, Array]:
    batch_size, num_cells = adjoint.shape[:2]
    num_dofs = qd.shape[1]
    max_dof = tangent_local.shape[-1]
    matrix_rows = batch_size * num_cells * SPATIAL_DIM
    outputs = scalable_segment_recurrence(
        adjoint.reshape(matrix_rows, SPATIAL_DIM),
        tangent_local.reshape(matrix_rows, max_dof),
        link_velocity.reshape(matrix_rows, 1),
        step_velocity.reshape(matrix_rows, 1),
        tangent_velocity_dot.reshape(matrix_rows, 1),
        global_to_local,
        qd,
        jacobian.reshape(batch_size * SPATIAL_DIM, num_dofs),
        jacobian_dot_qd.reshape(batch_size * SPATIAL_DIM, 1),
        velocity.reshape(batch_size * SPATIAL_DIM, 1),
        gravity.reshape(batch_size * SPATIAL_DIM, 1),
        output_dims={
            "jacobian_states": (matrix_rows, num_dofs),
            "jacobian_dot_qd_states": (matrix_rows, 1),
            "velocity_states": (matrix_rows, 1),
            "gravity_states": (matrix_rows, 1),
        },
    )
    return (
        outputs[0].reshape(
            batch_size, num_cells, SPATIAL_DIM, num_dofs
        ),
        outputs[1].reshape(batch_size, num_cells, SPATIAL_DIM),
        outputs[2].reshape(batch_size, num_cells, SPATIAL_DIM),
        outputs[3].reshape(batch_size, num_cells, SPATIAL_DIM),
    )


def _warp_scalable_assembly(
    jacobians: Array,
    jacobians_dot_qd: Array,
    velocities: Array,
    gravities: Array,
    weights: Array,
    mass_diagonals: Array,
) -> tuple[Array, Array, Array]:
    """Assemble batched GVS terms with one Warp owner per output entry."""

    batch_size, num_segments, num_quadrature, _, num_dofs = jacobians.shape
    state_rows = batch_size * num_segments * num_quadrature * SPATIAL_DIM
    outputs = scalable_assemble_dynamics(
        jacobians.reshape(state_rows, num_dofs),
        jacobians_dot_qd.reshape(state_rows, 1),
        velocities.reshape(state_rows, 1),
        gravities.reshape(state_rows, 1),
        weights.reshape(num_segments * num_quadrature),
        mass_diagonals.reshape(num_segments * num_quadrature, SPATIAL_DIM),
        jnp.asarray([num_quadrature], dtype=jnp.int32),
        output_dims={
            "inertia": (batch_size, num_dofs, num_dofs),
            "coriolis_qd": (batch_size, num_dofs),
            "gravity_force": (batch_size, num_dofs),
        },
    )
    return outputs[0], outputs[1], outputs[2]


def _warp_scalable_cooperative_segment(
    adjoint: Array,
    tangent_local: Array,
    link_velocity: Array,
    step_velocity: Array,
    tangent_velocity_dot: Array,
    global_to_local: Array,
    qd: Array,
    jacobian: Array,
    jacobian_dot_qd: Array,
    velocity: Array,
    gravity: Array,
    joint_adjoint: Array,
    joint_adjoint_dot: Array,
    joint_tangent: Array,
    joint_tangent_dot_qd: Array,
    joint_velocity: Array,
    apply_joint: bool,
    weights: Array,
    masses: Array,
    lanes_per_block: int,
) -> tuple[Array, ...]:
    """Advance and assemble one runtime-shaped segment per environment."""

    batch_size, num_cells = adjoint.shape[:2]
    num_dofs = qd.shape[1]
    max_dof = tangent_local.shape[-1]
    state_rows = batch_size * SPATIAL_DIM
    cell_rows = batch_size * num_cells * SPATIAL_DIM
    outputs = scalable_cooperative_segment(
        adjoint.reshape(cell_rows, SPATIAL_DIM),
        tangent_local.reshape(cell_rows, max_dof),
        link_velocity.reshape(cell_rows, 1),
        step_velocity.reshape(cell_rows, 1),
        tangent_velocity_dot.reshape(cell_rows, 1),
        global_to_local,
        qd,
        jacobian.reshape(state_rows, num_dofs),
        jacobian_dot_qd.reshape(state_rows, 1),
        velocity.reshape(state_rows, 1),
        gravity.reshape(state_rows, 1),
        joint_adjoint.reshape(state_rows, SPATIAL_DIM),
        joint_adjoint_dot.reshape(state_rows, SPATIAL_DIM),
        joint_tangent.reshape(state_rows, num_dofs),
        joint_tangent_dot_qd.reshape(state_rows, 1),
        joint_velocity.reshape(state_rows, 1),
        jnp.asarray([int(apply_joint)], dtype=jnp.int32),
        weights,
        masses,
        jnp.asarray([lanes_per_block], dtype=jnp.int32),
        output_dims={
            "jacobian_tip": (state_rows, num_dofs),
            "jacobian_dot_qd_tip": (state_rows, 1),
            "velocity_tip": (state_rows, 1),
            "gravity_tip": (state_rows, 1),
            "inertia": (batch_size, num_dofs, num_dofs),
            "coriolis_qd": (batch_size, num_dofs),
            "gravity_force": (batch_size, num_dofs),
            "jacobian_scratch": (state_rows, num_dofs),
            "jacobian_dot_qd_scratch": (state_rows, 1),
            "velocity_scratch": (state_rows, 1),
            "gravity_scratch": (state_rows, 1),
        },
    )
    return (
        outputs[0].reshape(batch_size, SPATIAL_DIM, num_dofs),
        outputs[1].reshape(batch_size, SPATIAL_DIM),
        outputs[2].reshape(batch_size, SPATIAL_DIM),
        outputs[3].reshape(batch_size, SPATIAL_DIM),
        outputs[4],
        outputs[5],
        outputs[6],
    )


def _warp_scalable_persistent_chain(
    joint_adjoint: Array,
    joint_adjoint_dot: Array,
    joint_tangent: Array,
    joint_tangent_dot_qd: Array,
    joint_velocity: Array,
    cell_adjoint: Array,
    cell_tangent_local: Array,
    cell_link_velocity: Array,
    cell_step_velocity: Array,
    cell_tangent_velocity_dot: Array,
    global_to_local: Array,
    active_dofs: Array,
    qd: Array,
    weights: Array,
    masses: Array,
    gravity_base: Array,
    lanes_per_block: int,
) -> tuple[Array, Array, Array]:
    """Evaluate a complete runtime-shaped serial GVS chain in one kernel."""

    batch_size, num_segments = joint_adjoint.shape[:2]
    num_cells = cell_adjoint.shape[2]
    num_quadrature = weights.shape[1]
    num_dofs = qd.shape[1]
    max_dof = cell_tangent_local.shape[-1]
    state_rows = batch_size * SPATIAL_DIM
    joint_rows = batch_size * num_segments * SPATIAL_DIM
    cell_rows = batch_size * num_segments * num_cells * SPATIAL_DIM
    state_output_dims = {
        "jacobian_first": (state_rows, num_dofs),
        "jacobian_dot_qd_first": (state_rows, 1),
        "velocity_first": (state_rows, 1),
        "gravity_first": (state_rows, 1),
        "jacobian_second": (state_rows, num_dofs),
        "jacobian_dot_qd_second": (state_rows, 1),
        "velocity_second": (state_rows, 1),
        "gravity_second": (state_rows, 1),
    }
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
        global_to_local,
        active_dofs,
        qd,
        weights.reshape(num_segments * num_quadrature),
        masses.reshape(num_segments * num_quadrature, SPATIAL_DIM),
        gravity_base,
        jnp.asarray([num_cells], dtype=jnp.int32),
        jnp.asarray([num_quadrature], dtype=jnp.int32),
        jnp.asarray([lanes_per_block], dtype=jnp.int32),
        output_dims={
            **state_output_dims,
            "inertia": (batch_size, num_dofs, num_dofs),
            "coriolis_qd": (batch_size, num_dofs),
            "gravity_force": (batch_size, num_dofs),
        },
    )
    return outputs[-3], outputs[-2], outputs[-1]


def _build_jax_forward_dynamics(
    terms: Callable[[Array, Array], tuple[Array, Array, Array]],
) -> Callable[[Array, Array], tuple[Array]]:
    """Add a zero-applied-force JAX dense solve to a dynamics-terms path."""

    def forward(q: Array, qd: Array) -> tuple[Array]:
        inertia, coriolis_qd, gravity_force = terms(q, qd)
        right_hand_side = -(coriolis_qd + gravity_force)
        return (
            jnp.linalg.solve(inertia, right_hand_side[..., None]).squeeze(-1),
        )

    return forward


def _build_warp_forward_dynamics(
    terms: Callable[[Array, Array], tuple[Array, Array, Array]],
    *,
    lanes_per_block: int,
) -> Callable[[Array, Array], tuple[Array]]:
    """Add the runtime-sized Warp Cholesky solve to a terms path."""

    def forward(q: Array, qd: Array) -> tuple[Array]:
        inertia, coriolis_qd, gravity_force = terms(q, qd)
        batch_size, num_dofs = coriolis_qd.shape
        outputs = scalable_cholesky_solve(
            inertia,
            coriolis_qd,
            gravity_force,
            jnp.asarray([lanes_per_block], dtype=jnp.int32),
            output_dims={
                "factor": (batch_size, num_dofs, num_dofs),
                "intermediate": (batch_size, num_dofs),
                "acceleration": (batch_size, num_dofs),
            },
        )
        return (outputs[-1],)

    return forward


def _warp_contract(
    tangent: Array,
    tangent_dot: Array,
    adjoint: Array,
    magnus_basis: Array,
    magnus_basis_dot: Array,
    qd_link: Array,
    gather_indices: Array,
    gather_mask: Array,
) -> tuple[Array, Array, Array, Array]:
    batch_size = tangent.shape[0]
    work_items = batch_size * NUM_SEGMENTS * NUM_CELLS
    contracted = contract_cell_terms(
        tangent.reshape(work_items, SPATIAL_DIM, SPATIAL_DIM),
        tangent_dot.reshape(work_items, SPATIAL_DIM, SPATIAL_DIM),
        adjoint.reshape(work_items, SPATIAL_DIM, SPATIAL_DIM),
        magnus_basis.reshape(work_items, SPATIAL_DIM, MAX_DOF),
        magnus_basis_dot.reshape(work_items, SPATIAL_DIM, MAX_DOF),
        qd_link,
        gather_indices,
        gather_mask,
        output_dims={
            "tangent_active": (
                work_items,
                SPATIAL_DIM,
                NUM_DOFS,
            ),
            "link_velocity": (work_items, SPATIAL_DIM),
            "step_velocity": (work_items, SPATIAL_DIM),
            "tangent_velocity_dot": (work_items, SPATIAL_DIM),
        },
    )
    return (
        contracted[0].reshape(
            batch_size,
            NUM_SEGMENTS,
            NUM_CELLS,
            SPATIAL_DIM,
            NUM_DOFS,
        ),
        *(
            value.reshape(batch_size, NUM_SEGMENTS, NUM_CELLS, SPATIAL_DIM)
            for value in contracted[1:]
        ),
    )


def _build_option_2(
    robot: Any,
    *,
    warp_contraction: bool = False,
    matrix_free_lie: bool = False,
    serial_lie: bool = False,
    constant_strain: bool = False,
    fixed_joints: bool = False,
) -> Callable[[Array, Array], tuple[Array, ...]]:
    if warp_contraction and matrix_free_lie:
        raise ValueError("Warp contraction and matrix-free Lie prep are exclusive")
    precompute = (
        _build_matrix_free_precompute(
            robot,
            serial_cell=serial_lie,
            constant_strain=constant_strain,
            fixed_joints=fixed_joints,
        )
        if matrix_free_lie
        else _build_precompute(robot)
    )
    mass_diagonals = jnp.diagonal(robot.inner_mass_matrices, axis1=-2, axis2=-1)
    gravity_initial = se3.adjoint_inverse(robot.g0) @ robot.g

    def dynamics(q: Array, qd: Array) -> tuple[Array, Array, Array]:
        (
            joint_adjoint,
            joint_adjoint_dot,
            joint_tangent,
            joint_tangent_dot_qd,
            joint_velocity,
            cell_adjoint,
            cell_tangent,
            cell_link_velocity,
            cell_step_velocity,
            cell_tangent_velocity_dot,
            cell_tangent_raw,
            cell_tangent_dot_raw,
            cell_magnus_basis_raw,
            cell_magnus_basis_dot_raw,
            qd_link,
        ) = precompute(q, qd)
        if warp_contraction:
            (
                cell_tangent,
                cell_link_velocity,
                cell_step_velocity,
                cell_tangent_velocity_dot,
            ) = _warp_contract(
                cell_tangent_raw,
                cell_tangent_dot_raw,
                cell_adjoint,
                cell_magnus_basis_raw,
                cell_magnus_basis_dot_raw,
                qd_link,
                jnp.asarray(robot.gather_indices[:, 1], dtype=jnp.int32),
                jnp.asarray(robot.gather_mask[:, 1], dtype=q.dtype),
            )
        batch_size = q.shape[0]
        jacobian = jnp.zeros((batch_size, SPATIAL_DIM, NUM_DOFS), dtype=q.dtype)
        vector = jnp.zeros((batch_size, SPATIAL_DIM), dtype=q.dtype)
        jacobian_dot_qd = vector
        velocity = vector
        gravity = jnp.broadcast_to(gravity_initial, (batch_size, SPATIAL_DIM))
        inertia = jnp.zeros((batch_size, NUM_DOFS, NUM_DOFS), dtype=q.dtype)
        coriolis_qd = jnp.zeros((batch_size, NUM_DOFS), dtype=q.dtype)
        gravity_force = jnp.zeros((batch_size, NUM_DOFS), dtype=q.dtype)

        for segment in range(NUM_SEGMENTS):
            adjoint = joint_adjoint[:, segment]
            jacobian_tip_qd = jnp.einsum("bij,bj->bi", jacobian, qd)
            jacobian = jnp.einsum(
                "bij,bjk->bik", adjoint, jacobian + joint_tangent[:, segment]
            )
            jacobian_dot_qd = jnp.einsum(
                "bij,bj->bi",
                adjoint,
                jacobian_dot_qd + joint_tangent_dot_qd[:, segment],
            ) + jnp.einsum("bij,bj->bi", joint_adjoint_dot[:, segment], jacobian_tip_qd)
            velocity = jnp.einsum(
                "bij,bj->bi", adjoint, velocity + joint_velocity[:, segment]
            )
            gravity = jnp.einsum("bij,bj->bi", adjoint, gravity)

            (
                jacobian_states,
                jacobian_dot_qd_states,
                velocity_states,
                gravity_states,
            ) = _warp_segment(
                cell_adjoint[:, segment],
                cell_tangent[:, segment],
                cell_link_velocity[:, segment],
                cell_step_velocity[:, segment],
                cell_tangent_velocity_dot[:, segment],
                qd,
                jacobian,
                jacobian_dot_qd,
                velocity,
                gravity,
            )
            jacobian = jacobian_states[:, -1]
            jacobian_dot_qd = jacobian_dot_qd_states[:, -1]
            velocity = velocity_states[:, -1]
            gravity = gravity_states[:, -1]

            jacobians = jacobian_states[:, :NUM_QUADRATURE_CELLS]
            jacobians_dot_qd = jacobian_dot_qd_states[:, :NUM_QUADRATURE_CELLS]
            velocities = velocity_states[:, :NUM_QUADRATURE_CELLS]
            gravities = gravity_states[:, :NUM_QUADRATURE_CELLS]
            weights = robot.inner_integration_weights[segment]
            masses = mass_diagonals[segment]
            weighted_masses = weights[:, None] * masses
            inertia += jnp.einsum(
                "bqri,qr,bqrj->bij", jacobians, weighted_masses, jacobians
            )
            momentum = masses[None, :, :] * velocities
            wrench = masses[None, :, :] * jacobians_dot_qd + _coadjoint_action(
                velocities, momentum
            )
            coriolis_qd += jnp.einsum(
                "bqri,bqr->bi", jacobians, weights[None, :, None] * wrench
            )
            gravity_force -= jnp.einsum(
                "bqri,bqr->bi",
                jacobians,
                weighted_masses[None, :, :] * gravities,
            )

        return inertia, coriolis_qd, gravity_force

    return dynamics


def _build_scalable_option(
    robot: Any,
    *,
    order_zero: bool = False,
    fixed_joints: bool = False,
    warp_joints: bool = False,
    warp_assembly: bool = False,
    cooperative_segment: bool = False,
    cooperative_lanes: int = 128,
    integrated_joint_segment: bool = False,
    persistent_chain: bool = False,
    active_prefix: bool = False,
) -> Callable[[Array, Array], tuple[Array, ...]]:
    """Runtime-shaped general GVS Lie and segment recurrence path."""

    precompute = _build_scalable_precompute(
        robot,
        order_zero=order_zero,
        fixed_joints=fixed_joints,
        warp_joints=warp_joints,
    )
    num_segments = int(robot.B_Z1.shape[0])
    num_cells = int(robot.B_Z1.shape[1])
    num_dofs = int(robot.num_dofs)
    mass_diagonals = jnp.diagonal(robot.inner_mass_matrices, axis1=-2, axis2=-1)
    gravity_initial = se3.adjoint_inverse(robot.g0) @ robot.g
    global_to_local_numpy = np.full(
        (num_segments, num_dofs), -1, dtype=np.int32
    )
    gather_indices_numpy = np.asarray(robot.gather_indices[:, 1])
    gather_mask_numpy = np.asarray(robot.gather_mask[:, 1])
    for segment in range(num_segments):
        for local_column in range(gather_indices_numpy.shape[1]):
            if gather_mask_numpy[segment, local_column]:
                global_column = gather_indices_numpy[segment, local_column]
                global_to_local_numpy[segment, global_column] = local_column
    global_to_local = jnp.asarray(global_to_local_numpy)
    active_dofs_numpy = np.full(num_segments, num_dofs, dtype=np.int32)
    if active_prefix:
        active_indices: set[int] = set()
        full_gather_indices = np.asarray(robot.gather_indices)
        full_gather_mask = np.asarray(robot.gather_mask)
        for segment in range(num_segments):
            for item in range(full_gather_indices.shape[1]):
                for local_column in range(full_gather_indices.shape[2]):
                    if full_gather_mask[segment, item, local_column]:
                        active_indices.add(
                            int(
                                full_gather_indices[
                                    segment, item, local_column
                                ]
                            )
                        )
            prefix_size = len(active_indices)
            if active_indices == set(range(prefix_size)):
                active_dofs_numpy[segment] = prefix_size
    active_dofs = jnp.asarray(active_dofs_numpy)
    if warp_assembly and cooperative_segment:
        raise ValueError(
            "Separate Warp assembly and cooperative segment paths are exclusive"
        )
    if integrated_joint_segment and not cooperative_segment:
        raise ValueError(
            "Integrated joint propagation requires the cooperative segment path"
        )
    if persistent_chain and (warp_assembly or cooperative_segment):
        raise ValueError(
            "Persistent-chain and per-segment assembly paths are exclusive"
        )

    def dynamics(q: Array, qd: Array) -> tuple[Array, Array, Array]:
        (
            joint_adjoint,
            joint_adjoint_dot,
            joint_tangent,
            joint_tangent_dot_qd,
            joint_velocity,
            cell_adjoint,
            cell_tangent_local,
            cell_link_velocity,
            cell_step_velocity,
            cell_tangent_velocity_dot,
        ) = precompute(q, qd)
        batch_size = q.shape[0]
        jacobian = jnp.zeros(
            (batch_size, SPATIAL_DIM, num_dofs), dtype=q.dtype
        )
        vector = jnp.zeros((batch_size, SPATIAL_DIM), dtype=q.dtype)
        jacobian_dot_qd = vector
        velocity = vector
        gravity = jnp.broadcast_to(gravity_initial, (batch_size, SPATIAL_DIM))
        inertia = jnp.zeros((batch_size, num_dofs, num_dofs), dtype=q.dtype)
        coriolis_qd = jnp.zeros((batch_size, num_dofs), dtype=q.dtype)
        gravity_force = jnp.zeros((batch_size, num_dofs), dtype=q.dtype)
        jacobian_quadrature = []
        jacobian_dot_qd_quadrature = []
        velocity_quadrature = []
        gravity_quadrature = []

        if persistent_chain:
            return _warp_scalable_persistent_chain(
                joint_adjoint,
                joint_adjoint_dot,
                joint_tangent,
                joint_tangent_dot_qd,
                joint_velocity,
                cell_adjoint,
                cell_tangent_local,
                cell_link_velocity,
                cell_step_velocity,
                cell_tangent_velocity_dot,
                global_to_local,
                active_dofs,
                qd,
                robot.inner_integration_weights,
                mass_diagonals,
                gravity_initial,
                cooperative_lanes,
            )

        for segment in range(num_segments):
            adjoint = joint_adjoint[:, segment]
            if integrated_joint_segment:
                (
                    jacobian,
                    jacobian_dot_qd,
                    velocity,
                    gravity,
                    segment_inertia,
                    segment_coriolis_qd,
                    segment_gravity_force,
                ) = _warp_scalable_cooperative_segment(
                    cell_adjoint[:, segment],
                    cell_tangent_local[:, segment],
                    cell_link_velocity[:, segment],
                    cell_step_velocity[:, segment],
                    cell_tangent_velocity_dot[:, segment],
                    global_to_local[segment],
                    qd,
                    jacobian,
                    jacobian_dot_qd,
                    velocity,
                    gravity,
                    joint_adjoint[:, segment],
                    joint_adjoint_dot[:, segment],
                    joint_tangent[:, segment],
                    joint_tangent_dot_qd[:, segment],
                    joint_velocity[:, segment],
                    True,
                    robot.inner_integration_weights[segment],
                    mass_diagonals[segment],
                    cooperative_lanes,
                )
                inertia += segment_inertia
                coriolis_qd += segment_coriolis_qd
                gravity_force += segment_gravity_force
                continue
            jacobian_tip_qd = jnp.einsum("bij,bj->bi", jacobian, qd)
            jacobian = jnp.einsum(
                "bij,bjk->bik", adjoint, jacobian + joint_tangent[:, segment]
            )
            jacobian_dot_qd = jnp.einsum(
                "bij,bj->bi",
                adjoint,
                jacobian_dot_qd + joint_tangent_dot_qd[:, segment],
            ) + jnp.einsum(
                "bij,bj->bi", joint_adjoint_dot[:, segment], jacobian_tip_qd
            )
            velocity = jnp.einsum(
                "bij,bj->bi", adjoint, velocity + joint_velocity[:, segment]
            )
            gravity = jnp.einsum("bij,bj->bi", adjoint, gravity)

            if cooperative_segment:
                (
                    jacobian,
                    jacobian_dot_qd,
                    velocity,
                    gravity,
                    segment_inertia,
                    segment_coriolis_qd,
                    segment_gravity_force,
                ) = _warp_scalable_cooperative_segment(
                    cell_adjoint[:, segment],
                    cell_tangent_local[:, segment],
                    cell_link_velocity[:, segment],
                    cell_step_velocity[:, segment],
                    cell_tangent_velocity_dot[:, segment],
                    global_to_local[segment],
                    qd,
                    jacobian,
                    jacobian_dot_qd,
                    velocity,
                    gravity,
                    adjoint,
                    adjoint,
                    jacobian,
                    jacobian_dot_qd,
                    velocity,
                    False,
                    robot.inner_integration_weights[segment],
                    mass_diagonals[segment],
                    cooperative_lanes,
                )
                inertia += segment_inertia
                coriolis_qd += segment_coriolis_qd
                gravity_force += segment_gravity_force
                continue

            (
                jacobian_states,
                jacobian_dot_qd_states,
                velocity_states,
                gravity_states,
            ) = _warp_scalable_segment(
                cell_adjoint[:, segment],
                cell_tangent_local[:, segment],
                cell_link_velocity[:, segment],
                cell_step_velocity[:, segment],
                cell_tangent_velocity_dot[:, segment],
                global_to_local[segment],
                qd,
                jacobian,
                jacobian_dot_qd,
                velocity,
                gravity,
            )
            jacobian = jacobian_states[:, -1]
            jacobian_dot_qd = jacobian_dot_qd_states[:, -1]
            velocity = velocity_states[:, -1]
            gravity = gravity_states[:, -1]

            jacobians = jacobian_states[:, : num_cells - 1]
            jacobians_dot_qd = jacobian_dot_qd_states[:, : num_cells - 1]
            velocities = velocity_states[:, : num_cells - 1]
            gravities = gravity_states[:, : num_cells - 1]
            if warp_assembly:
                jacobian_quadrature.append(jacobians)
                jacobian_dot_qd_quadrature.append(jacobians_dot_qd)
                velocity_quadrature.append(velocities)
                gravity_quadrature.append(gravities)
                continue
            weights = robot.inner_integration_weights[segment]
            masses = mass_diagonals[segment]
            weighted_masses = weights[:, None] * masses
            inertia += jnp.einsum(
                "bqri,qr,bqrj->bij", jacobians, weighted_masses, jacobians
            )
            momentum = masses[None, :, :] * velocities
            wrench = masses[None, :, :] * jacobians_dot_qd + _coadjoint_action(
                velocities, momentum
            )
            coriolis_qd += jnp.einsum(
                "bqri,bqr->bi", jacobians, weights[None, :, None] * wrench
            )
            gravity_force -= jnp.einsum(
                "bqri,bqr->bi",
                jacobians,
                weighted_masses[None, :, :] * gravities,
            )

        if warp_assembly:
            return _warp_scalable_assembly(
                jnp.stack(jacobian_quadrature, axis=1),
                jnp.stack(jacobian_dot_qd_quadrature, axis=1),
                jnp.stack(velocity_quadrature, axis=1),
                jnp.stack(gravity_quadrature, axis=1),
                robot.inner_integration_weights,
                mass_diagonals,
            )
        return inertia, coriolis_qd, gravity_force

    return dynamics


def _warp_cooperative_segment(
    adjoint: Array,
    tangent_active: Array,
    link_velocity: Array,
    step_velocity: Array,
    tangent_velocity_dot: Array,
    qd: Array,
    jacobian: Array,
    jacobian_dot_qd: Array,
    velocity: Array,
    gravity: Array,
    weights: Array,
    masses: Array,
) -> tuple[Array, ...]:
    batch_size = qd.shape[0]
    matrix_rows = batch_size * NUM_CELLS * SPATIAL_DIM
    vector_rows = batch_size * NUM_CELLS * SPATIAL_DIM
    state_rows = batch_size * NUM_CELLS * SPATIAL_DIM
    outputs = cooperative_segment_dynamics(
        adjoint.reshape(matrix_rows, SPATIAL_DIM),
        tangent_active.reshape(matrix_rows, NUM_DOFS),
        link_velocity.reshape(vector_rows, 1),
        step_velocity.reshape(vector_rows, 1),
        tangent_velocity_dot.reshape(vector_rows, 1),
        qd.reshape(batch_size * NUM_DOFS, 1),
        jacobian.reshape(batch_size * SPATIAL_DIM, NUM_DOFS),
        jacobian_dot_qd.reshape(batch_size * SPATIAL_DIM, 1),
        velocity.reshape(batch_size * SPATIAL_DIM, 1),
        gravity.reshape(batch_size * SPATIAL_DIM, 1),
        weights,
        masses.reshape(NUM_QUADRATURE_CELLS * SPATIAL_DIM, 1),
        output_dims={
            "jacobian_states": (state_rows, NUM_DOFS),
            "jacobian_dot_qd_states": (state_rows, 1),
            "velocity_states": (state_rows, 1),
            "gravity_states": (state_rows, 1),
            "inertia": (batch_size * NUM_DOFS, NUM_DOFS),
            "coriolis_qd": (batch_size * NUM_DOFS, 1),
            "gravity_force": (batch_size * NUM_DOFS, 1),
        },
    )
    jacobian_states = outputs[0].reshape(batch_size, NUM_CELLS, SPATIAL_DIM, NUM_DOFS)
    jacobian_dot_qd_states = outputs[1].reshape(batch_size, NUM_CELLS, SPATIAL_DIM)
    velocity_states = outputs[2].reshape(batch_size, NUM_CELLS, SPATIAL_DIM)
    gravity_states = outputs[3].reshape(batch_size, NUM_CELLS, SPATIAL_DIM)
    return (
        jacobian_states[:, -1],
        jacobian_dot_qd_states[:, -1],
        velocity_states[:, -1],
        gravity_states[:, -1],
        outputs[4].reshape(batch_size, NUM_DOFS, NUM_DOFS),
        outputs[5].reshape(batch_size, NUM_DOFS),
        outputs[6].reshape(batch_size, NUM_DOFS),
    )


def _build_option_5(
    robot: Any,
    *,
    matrix_free_lie: bool = False,
    serial_lie: bool = False,
    constant_strain: bool = False,
    fixed_joints: bool = False,
) -> Callable[[Array, Array], tuple[Array, ...]]:
    """Cooperative one-block-per-environment segment recurrence and assembly."""

    precompute = (
        _build_matrix_free_precompute(
            robot,
            serial_cell=serial_lie,
            constant_strain=constant_strain,
            fixed_joints=fixed_joints,
        )
        if matrix_free_lie
        else _build_precompute(robot)
    )
    mass_diagonals = jnp.diagonal(robot.inner_mass_matrices, axis1=-2, axis2=-1)
    gravity_initial = se3.adjoint_inverse(robot.g0) @ robot.g

    def dynamics(q: Array, qd: Array) -> tuple[Array, Array, Array]:
        (
            joint_adjoint,
            joint_adjoint_dot,
            joint_tangent,
            joint_tangent_dot_qd,
            joint_velocity,
            cell_adjoint,
            cell_tangent,
            cell_link_velocity,
            cell_step_velocity,
            cell_tangent_velocity_dot,
            _cell_tangent_raw,
            _cell_tangent_dot_raw,
            _cell_magnus_basis_raw,
            _cell_magnus_basis_dot_raw,
            _qd_link,
        ) = precompute(q, qd)
        batch_size = q.shape[0]
        jacobian = jnp.zeros((batch_size, SPATIAL_DIM, NUM_DOFS), dtype=q.dtype)
        vector = jnp.zeros((batch_size, SPATIAL_DIM), dtype=q.dtype)
        jacobian_dot_qd = vector
        velocity = vector
        gravity = jnp.broadcast_to(gravity_initial, (batch_size, SPATIAL_DIM))
        inertia = jnp.zeros((batch_size, NUM_DOFS, NUM_DOFS), dtype=q.dtype)
        coriolis_qd = jnp.zeros((batch_size, NUM_DOFS), dtype=q.dtype)
        gravity_force = jnp.zeros((batch_size, NUM_DOFS), dtype=q.dtype)

        for segment in range(NUM_SEGMENTS):
            adjoint = joint_adjoint[:, segment]
            jacobian_tip_qd = jnp.einsum("bij,bj->bi", jacobian, qd)
            jacobian = jnp.einsum(
                "bij,bjk->bik", adjoint, jacobian + joint_tangent[:, segment]
            )
            jacobian_dot_qd = jnp.einsum(
                "bij,bj->bi",
                adjoint,
                jacobian_dot_qd + joint_tangent_dot_qd[:, segment],
            ) + jnp.einsum("bij,bj->bi", joint_adjoint_dot[:, segment], jacobian_tip_qd)
            velocity = jnp.einsum(
                "bij,bj->bi", adjoint, velocity + joint_velocity[:, segment]
            )
            gravity = jnp.einsum("bij,bj->bi", adjoint, gravity)

            (
                jacobian,
                jacobian_dot_qd,
                velocity,
                gravity,
                inertia_segment,
                coriolis_segment,
                gravity_segment,
            ) = _warp_cooperative_segment(
                cell_adjoint[:, segment],
                cell_tangent[:, segment],
                cell_link_velocity[:, segment],
                cell_step_velocity[:, segment],
                cell_tangent_velocity_dot[:, segment],
                qd,
                jacobian,
                jacobian_dot_qd,
                velocity,
                gravity,
                robot.inner_integration_weights[segment],
                mass_diagonals[segment],
            )
            inertia += inertia_segment
            coriolis_qd += coriolis_segment
            gravity_force += gravity_segment

        return inertia, coriolis_qd, gravity_force

    return dynamics


def _build_option_4(robot: Any) -> Callable[[Array, Array], tuple[Array, ...]]:
    precompute = _build_precompute(robot)
    mass_diagonals = jnp.diagonal(robot.inner_mass_matrices, axis1=-2, axis2=-1)
    gravity_initial = se3.adjoint_inverse(robot.g0) @ robot.g
    total_states = NUM_SEGMENTS * (NUM_CELLS + 1)

    def dynamics(q: Array, qd: Array) -> tuple[Array, Array, Array]:
        (
            joint_adjoint,
            joint_adjoint_dot,
            joint_tangent,
            joint_tangent_dot_qd,
            joint_velocity,
            cell_adjoint,
            _cell_tangent_active,
            _cell_link_velocity,
            _cell_step_velocity,
            _cell_tangent_velocity_dot,
            cell_tangent_raw,
            cell_tangent_dot_raw,
            cell_magnus_basis_raw,
            cell_magnus_basis_dot_raw,
            qd_link,
        ) = precompute(q, qd)
        batch_size = q.shape[0]
        joint_items = batch_size * NUM_SEGMENTS
        cell_items = batch_size * NUM_SEGMENTS * NUM_CELLS
        outputs = persistent_raw_dynamics(
            joint_adjoint.reshape(joint_items, SPATIAL_DIM, SPATIAL_DIM),
            joint_adjoint_dot.reshape(joint_items, SPATIAL_DIM, SPATIAL_DIM),
            joint_tangent.reshape(joint_items, SPATIAL_DIM, NUM_DOFS),
            joint_tangent_dot_qd.reshape(joint_items, SPATIAL_DIM),
            joint_velocity.reshape(joint_items, SPATIAL_DIM),
            cell_adjoint.reshape(cell_items, SPATIAL_DIM, SPATIAL_DIM),
            cell_tangent_raw.reshape(cell_items, SPATIAL_DIM, SPATIAL_DIM),
            cell_tangent_dot_raw.reshape(cell_items, SPATIAL_DIM, SPATIAL_DIM),
            cell_magnus_basis_raw.reshape(cell_items, SPATIAL_DIM, MAX_DOF),
            cell_magnus_basis_dot_raw.reshape(cell_items, SPATIAL_DIM, MAX_DOF),
            qd,
            qd_link,
            jnp.asarray(robot.gather_indices[:, 1], dtype=jnp.int32),
            jnp.asarray(robot.gather_mask[:, 1], dtype=q.dtype),
            robot.inner_integration_weights,
            mass_diagonals,
            gravity_initial,
            output_dims={
                "jacobian_states": (
                    batch_size,
                    total_states,
                    SPATIAL_DIM,
                    NUM_DOFS,
                ),
                "jacobian_dot_qd_states": (
                    batch_size,
                    total_states,
                    SPATIAL_DIM,
                ),
                "velocity_states": (batch_size, total_states, SPATIAL_DIM),
                "gravity_states": (batch_size, total_states, SPATIAL_DIM),
                "inertia": (batch_size, NUM_DOFS, NUM_DOFS),
                "coriolis_qd": (batch_size, NUM_DOFS),
                "gravity_force": (batch_size, NUM_DOFS),
            },
        )
        return outputs[-3:]

    return dynamics


def _make_inputs(
    robot: Any, batch_size: int, device: jax.Device
) -> tuple[Array, Array]:
    context = _gvs_context(robot)
    phase = jnp.linspace(0.0, 1.0, batch_size, dtype=jnp.float64)[:, None]
    q = context["q"][None, :] + 1.0e-3 * jnp.sin(phase)
    qd = context["qd"][None, :] + 1.0e-3 * jnp.cos(phase)
    return jax.device_put(q, device), jax.device_put(qd, device)


def _timed_many_ms(
    functions: dict[str, Callable[..., Tree]],
    args: tuple[Array, ...],
    repeats: int,
) -> dict[str, dict[str, float]]:
    compiled = {name: jax.jit(function) for name, function in functions.items()}
    for function in compiled.values():
        _block(function(*args))

    samples: dict[str, list[float]] = {name: [] for name in functions}
    names = list(functions)
    for repeat in range(repeats):
        offset = repeat % len(names)
        for name in names[offset:] + names[:offset]:
            start = time.perf_counter()
            result = compiled[name](*args)
            _block(result)
            samples[name].append((time.perf_counter() - start) * 1.0e3)

    return {
        name: {
            "median_ms": statistics.median(values),
            "p25_ms": float(np.quantile(values, 0.25)),
            "p75_ms": float(np.quantile(values, 0.75)),
        }
        for name, values in samples.items()
    }


def _cuda_profiler_library() -> ctypes.CDLL:
    candidates = [Path("libcudart.so.13"), Path("libcudart.so")]
    candidates.extend(
        sorted(
            Path(sys.prefix).glob(
                "lib/python*/site-packages/nvidia/cu*/lib/libcudart.so*"
            ),
            reverse=True,
        )
    )
    errors: list[str] = []
    for candidate in candidates:
        try:
            library = ctypes.CDLL(str(candidate))
        except OSError as error:
            errors.append(f"{candidate}: {error}")
            continue
        for function_name in ("cudaProfilerStart", "cudaProfilerStop"):
            function = getattr(library, function_name)
            function.argtypes = []
            function.restype = ctypes.c_int
        return library
    raise RuntimeError("Could not load libcudart:\n" + "\n".join(errors))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    option_names = (
        "baseline",
        "option_1",
        "option_2",
        "option_3",
        "option_4",
        "option_5",
        "option_6",
        "option_7",
        "option_8",
        "option_9",
        "option_10",
        "option_11",
        "option_12",
        "option_13",
        "option_14",
        "option_15",
        "option_16",
        "option_17",
        "option_18",
        "option_19",
        "option_20",
        "option_21",
        "option_22",
        "option_23",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--operation",
        choices=("terms", "forward"),
        default="terms",
        help="Benchmark M/C/G terms or zero-input forward dynamics.",
    )
    parser.add_argument("--segment-count", type=int, default=NUM_SEGMENTS)
    parser.add_argument("--basis-order", type=int, default=0)
    parser.add_argument("--joint-type", choices=JOINT_TYPES, default="fixed")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--options",
        choices=option_names,
        nargs="+",
        default=tuple(name for name in option_names if name != "option_22"),
        help="Implementations included in the interleaved timing comparison.",
    )
    parser.add_argument(
        "--profile-option",
        choices=option_names,
        help="Run only this compiled option inside a CUDA profiler API range.",
    )
    parser.add_argument("--profile-iterations", type=int, default=10)
    parser.add_argument(
        "--compile-option",
        choices=option_names,
        help="Measure uncached lowering and compilation time without benchmarking.",
    )
    parser.add_argument("--compile-repeats", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    wp.init()
    device = jax.devices(args.device)[0]
    if args.segment_count < 1 or args.basis_order < 0:
        raise ValueError("Segment count must be positive and basis order non-negative")
    if args.basis_order == 0 and args.joint_type == "fixed":
        robot = _gvs_factory(
            args.segment_count, gauss_points=NUM_QUADRATURE_CELLS
        )
    else:
        segments = [
            _gvs_segment(
                strain_basis_order=args.basis_order,
                gauss_points=NUM_QUADRATURE_CELLS,
            )
            for _ in range(args.segment_count)
        ]
        if args.joint_type == "mixed":
            mixed_types = JOINT_TYPES[1:-1]
            for index, segment in enumerate(segments):
                segment.joint = _joint_spec(
                    mixed_types[index % len(mixed_types)]
                )
        elif args.joint_type != "fixed":
            for segment in segments:
                segment.joint = _joint_spec(args.joint_type)
        robot = GVS.from_segments(
            segments=segments,
            gravity=jnp.array([0.0, 0.0, 9.81]),
        )

    if args.compile_option is not None:
        requested_names = {"baseline", args.compile_option}
    elif args.profile_option is not None:
        requested_names = {"baseline", args.profile_option}
    else:
        requested_names = {"baseline", *args.options}
    shape_generic_names = {
        "baseline",
        "option_15",
        "option_16",
        "option_17",
        "option_18",
        "option_19",
        "option_20",
        "option_21",
        "option_22",
        "option_23",
    }
    fixed_shape_names = requested_names - shape_generic_names
    if fixed_shape_names and (
        robot.num_dofs != NUM_DOFS or robot.max_dof != MAX_DOF
    ):
        raise ValueError(
            f"Options {sorted(fixed_shape_names)} require the original "
            f"{NUM_SEGMENTS}-segment/order-zero shape; use option_15 or "
            "option_17 for "
            "shape-generic GVS experiments."
        )
    baseline = jax.vmap(robot.dynamics_terms)
    builders: dict[str, Callable[[], Callable[[Array, Array], tuple[Array, ...]]]] = {
        "option_1": lambda: _build_option_1(robot),
        "option_2": lambda: _build_option_2(robot),
        "option_3": lambda: _build_option_2(robot, warp_contraction=True),
        "option_4": lambda: _build_option_4(robot),
        "option_5": lambda: _build_option_5(robot),
        "option_6": lambda: _build_option_2(robot, matrix_free_lie=True),
        "option_7": lambda: _build_option_5(robot, matrix_free_lie=True),
        "option_8": lambda: _build_option_2(
            robot, matrix_free_lie=True, serial_lie=True
        ),
        "option_9": lambda: _build_option_5(
            robot, matrix_free_lie=True, serial_lie=True
        ),
        "option_10": lambda: _build_option_2(
            robot, matrix_free_lie=True, constant_strain=True
        ),
        "option_11": lambda: _build_option_5(
            robot, matrix_free_lie=True, constant_strain=True
        ),
        "option_12": lambda: _build_option_2(
            robot,
            matrix_free_lie=True,
            serial_lie=True,
            fixed_joints=True,
        ),
        "option_13": lambda: _build_option_2(
            robot,
            matrix_free_lie=True,
            constant_strain=True,
            fixed_joints=True,
        ),
        "option_14": lambda: _build_option_5(
            robot,
            matrix_free_lie=True,
            constant_strain=True,
            fixed_joints=True,
        ),
        "option_15": lambda: _build_scalable_option(robot),
        "option_16": lambda: _build_scalable_option(
            robot,
            order_zero=True,
            fixed_joints=True,
        ),
        "option_17": lambda: _build_scalable_option(
            robot,
            warp_joints=True,
        ),
        "option_18": lambda: _build_scalable_option(
            robot,
            warp_joints=True,
            warp_assembly=True,
        ),
        "option_19": lambda: _build_scalable_option(
            robot,
            warp_joints=True,
            cooperative_segment=True,
            cooperative_lanes=1 if args.device == "cpu" else 128,
        ),
        "option_20": lambda: _build_scalable_option(
            robot,
            warp_joints=True,
            cooperative_segment=True,
            cooperative_lanes=1 if args.device == "cpu" else 128,
            integrated_joint_segment=True,
        ),
        "option_21": lambda: _build_scalable_option(
            robot,
            warp_joints=True,
            cooperative_lanes=1 if args.device == "cpu" else 128,
            persistent_chain=True,
        ),
        "option_23": lambda: _build_scalable_option(
            robot,
            warp_joints=True,
            cooperative_lanes=1 if args.device == "cpu" else 128,
            persistent_chain=True,
            active_prefix=True,
        ),
    }
    if "option_22" in requested_names and args.operation != "forward":
        raise ValueError("option_22 requires --operation forward")
    if args.operation == "forward":
        baseline = _build_jax_forward_dynamics(baseline)
        builders = {
            name: (
                lambda builder=builder: _build_jax_forward_dynamics(builder())
            )
            for name, builder in builders.items()
        }
        builders["option_22"] = lambda: _build_warp_forward_dynamics(
            _build_scalable_option(
                robot,
                warp_joints=True,
                cooperative_lanes=1 if args.device == "cpu" else 128,
                persistent_chain=True,
            ),
            lanes_per_block=1 if args.device == "cpu" else 128,
        )
    all_functions = {"baseline": baseline}
    all_functions.update(
        {name: builders[name]() for name in requested_names - {"baseline"}}
    )
    selected_names = [
        name
        for name in dict.fromkeys(("baseline", *args.options))
        if name in all_functions
    ]
    functions = {name: all_functions[name] for name in selected_names}

    if args.compile_option is not None:
        if args.compile_repeats < 1:
            raise ValueError("--compile-repeats must be positive")
        compile_records: list[dict[str, Any]] = []
        function = all_functions[args.compile_option]
        for batch_size in args.batch_sizes:
            q, qd = _make_inputs(robot, batch_size, device)
            lower_samples: list[float] = []
            compile_samples: list[float] = []
            first_call_samples: list[float] = []
            for _ in range(args.compile_repeats):
                jax.clear_caches()
                jitted = jax.jit(function)

                start = time.perf_counter()
                lowered = jitted.lower(q, qd)
                lower_samples.append((time.perf_counter() - start) * 1.0e3)

                start = time.perf_counter()
                compiled = lowered.compile()
                compile_samples.append((time.perf_counter() - start) * 1.0e3)

                start = time.perf_counter()
                _block(compiled(q, qd))
                first_call_samples.append((time.perf_counter() - start) * 1.0e3)

            record = {
                "device": str(device),
                "batch_size": batch_size,
                "compile_option": args.compile_option,
                "repeats": args.compile_repeats,
                "lower_ms": statistics.median(lower_samples),
                "compile_ms": statistics.median(compile_samples),
                "total_lower_compile_ms": statistics.median(
                    [
                        lower + compile_time
                        for lower, compile_time in zip(
                            lower_samples, compile_samples, strict=True
                        )
                    ]
                ),
                "first_call_ms": statistics.median(first_call_samples),
                "samples": {
                    "lower_ms": lower_samples,
                    "compile_ms": compile_samples,
                    "first_call_ms": first_call_samples,
                },
            }
            compile_records.append(record)
            print(json.dumps(record, sort_keys=True))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(compile_records, indent=2) + "\n")
        return

    if args.profile_option is not None:
        if args.device != "gpu":
            raise ValueError("--profile-option requires --device gpu")
        batch_size = args.batch_sizes[0]
        q, qd = _make_inputs(robot, batch_size, device)
        compiled = jax.jit(all_functions[args.profile_option]).lower(q, qd).compile()
        _block(compiled(q, qd))
        cuda_runtime = _cuda_profiler_library()
        if cuda_runtime.cudaProfilerStart() != 0:
            raise RuntimeError("cudaProfilerStart failed")
        result = None
        for _ in range(args.profile_iterations):
            result = compiled(q, qd)
        assert result is not None
        _block(result)
        if cuda_runtime.cudaProfilerStop() != 0:
            raise RuntimeError("cudaProfilerStop failed")
        print(
            json.dumps(
                {
                    "profile_option": args.profile_option,
                    "batch_size": batch_size,
                    "iterations": args.profile_iterations,
                    "device": str(device),
                },
                sort_keys=True,
            )
        )
        return
    records: list[dict[str, Any]] = []

    for batch_size in args.batch_sizes:
        q, qd = _make_inputs(robot, batch_size, device)
        with jax.default_device(device):
            results = {
                name: jax.jit(function)(q, qd) for name, function in functions.items()
            }
            _block(results)
            baseline_result = results["baseline"]
            errors = {}
            for name, actual_result in results.items():
                if name == "baseline":
                    continue
                errors[name] = [
                    float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
                    for actual, expected in zip(
                        actual_result, baseline_result, strict=True
                    )
                ]
                tolerance = 2.0e-6 if args.operation == "forward" else 2.0e-9
                if max(errors[name]) > tolerance:
                    raise AssertionError(
                        f"{name} correctness failure: max errors {errors[name]}"
                    )
            timings = _timed_many_ms(functions, (q, qd), args.repeats)
        baseline_ms = timings["baseline"]["median_ms"]
        record = {
            "device": str(device),
            "operation": args.operation,
            "batch_size": batch_size,
            "baseline_ms": baseline_ms,
            "timing_distributions": timings,
            "max_abs_errors": errors,
        }
        for name, distribution in timings.items():
            if name == "baseline":
                continue
            option_ms = distribution["median_ms"]
            record[f"{name}_ms"] = option_ms
            record[f"{name}_speedup"] = baseline_ms / option_ms
        records.append(record)
        print(json.dumps(record, sort_keys=True))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
