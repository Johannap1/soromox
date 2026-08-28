"""JAX-facing fused Warp kinematics executor for PlanarPCS and PCS."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from soromox.systems.execution.types import KinematicsOperation, KinematicsResult
from soromox.systems.execution.warp.pcs.operands import (
    PCSKinematicsOperands,
    PCSKinematicsShapes,
)
from soromox.systems.execution.warp.pcs.planar_kinematics import (
    planar_forward_kinematics,
    planar_inertial_jacobians,
    planar_kinematics,
)
from soromox.systems.execution.warp.pcs.spatial_kinematics import (
    spatial_cooperative_inertial_jacobians,
    spatial_cooperative_kinematics,
    spatial_forward_kinematics,
    spatial_inertial_jacobians,
    spatial_kinematics,
)


def execute_kinematics(
    operands: PCSKinematicsOperands,
    q: Array,
    sample_s: Array,
    operation: KinematicsOperation,
) -> KinematicsResult:
    """Execute batched PCS poses and inertial-frame Jacobians in Warp.

    Args:
        operands: Runtime PCS or PlanarPCS model data.
        q: Batched active configurations with shape ``(E, D)``.
        sample_s: Per-environment abscissae with shape ``(E, N)``.
        operation: Select poses, inertial Jacobians, or both.

    Returns:
        Poses, Jacobians, or their tuple with canonical ``(E, N, ...)``
        leading dimensions.
    """

    shapes = PCSKinematicsShapes.from_operands(
        operands,
        batch_size=q.shape[0],
        num_samples=sample_s.shape[1],
    )
    common = (
        q,
        sample_s,
        operands.active_strain_indices,
        operands.active_strain_scales,
        operands.reference_strain.reshape(
            operands.num_segments, 3 if operands.is_planar else 6
        ),
        operands.segment_lengths,
        operands.segment_starts,
        operands.base_pose,
    )
    if operands.is_planar:
        epsilons = jnp.asarray(
            [operands.global_eps, operands.tangent_eps], dtype=jnp.float64
        )
        if operation == "pose":
            output_dims = shapes.pose_workspace()
            output_dims["poses"] = shapes.pose_output()
            return planar_forward_kinematics(
                *common, epsilons, output_dims=output_dims
            )[-1]
        if operation == "jacobian":
            output_dims = shapes.jacobian_workspace()
            output_dims["jacobians"] = shapes.jacobian_output()
            return planar_inertial_jacobians(
                *common, epsilons, output_dims=output_dims
            )[-1]
        output_dims = shapes.workspace()
        output_dims.update(
            {
                "poses": shapes.pose_output(),
                "jacobians": shapes.jacobian_output(),
            }
        )
        outputs = planar_kinematics(*common, epsilons, output_dims=output_dims)
    else:
        if operation == "pose":
            output_dims = shapes.pose_workspace()
            output_dims["poses"] = shapes.pose_output()
            return spatial_forward_kinematics(*common, output_dims=output_dims)[-1]
        if operation == "jacobian":
            output_dims = shapes.jacobian_workspace()
            output_dims["jacobians"] = shapes.jacobian_output()
            evaluator = (
                spatial_cooperative_inertial_jacobians
                if jax.default_backend() == "gpu"
                else spatial_inertial_jacobians
            )
            return evaluator(*common, output_dims=output_dims)[-1]
        output_dims = shapes.workspace()
        output_dims.update(
            {
                "poses": shapes.pose_output(),
                "jacobians": shapes.jacobian_output(),
            }
        )
        evaluator = (
            spatial_cooperative_kinematics
            if jax.default_backend() == "gpu"
            else spatial_kinematics
        )
        outputs = evaluator(*common, output_dims=output_dims)
    poses, jacobians = outputs[-2:]
    if operation == "pose":
        return poses
    if operation == "jacobian":
        return jacobians
    return poses, jacobians


__all__ = ["execute_kinematics"]
