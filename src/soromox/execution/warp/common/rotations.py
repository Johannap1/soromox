# ruff: noqa: I001, UP018
"""Shared quaternion rotation operators for continuum Warp kernels."""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.func
def _load_normalized_quaternion(
    base_pose: wp.array2d[wp.float64], environment: int
) -> wp.vec4d:
    """Load one normalized scalar-last quaternion from a batched base pose.

    Args:
        base_pose: Batched spatial poses with shape ``(E, 7)`` in
            ``[qx, qy, qz, qw, x, y, z]`` order.
        environment: Environment row in ``base_pose``.

    Returns:
        Unit Hamilton quaternion in scalar-last order. An exactly zero
        quaternion maps to ``[0, 0, 0, 1]`` to match the trace-safe JAX path.
    """
    x = base_pose[environment, 0]
    y = base_pose[environment, 1]
    z = base_pose[environment, 2]
    w = base_pose[environment, 3]
    norm_sq = x * x + y * y + z * z + w * w
    if norm_sq <= wp.float64(0.0):
        return wp.vec4d(
            wp.float64(0.0),
            wp.float64(0.0),
            wp.float64(0.0),
            wp.float64(1.0),
        )
    inverse_norm = wp.float64(1.0) / wp.sqrt(norm_sq)
    return wp.vec4d(
        x * inverse_norm,
        y * inverse_norm,
        z * inverse_norm,
        w * inverse_norm,
    )


@wp.func
def _quaternion_rotation_matrix(
    base_pose: wp.array2d[wp.float64], environment: int
) -> wp.mat33d:
    """Return one normalized scalar-last quaternion rotation matrix.

    Args:
        base_pose: Batched spatial poses with shape ``(E, 7)`` in
            ``[qx, qy, qz, qw, x, y, z]`` order.
        environment: Environment row in ``base_pose``.

    Returns:
        The normalized three-dimensional rotation. A zero quaternion follows
        the trace-safe JAX path and maps to identity.
    """
    quaternion = _load_normalized_quaternion(base_pose, environment)
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]
    rotation = wp.mat33d()
    rotation[0, 0] = wp.float64(1.0) - wp.float64(2.0) * (y * y + z * z)
    rotation[0, 1] = wp.float64(2.0) * (x * y - w * z)
    rotation[0, 2] = wp.float64(2.0) * (x * z + w * y)
    rotation[1, 0] = wp.float64(2.0) * (x * y + w * z)
    rotation[1, 1] = wp.float64(1.0) - wp.float64(2.0) * (x * x + z * z)
    rotation[1, 2] = wp.float64(2.0) * (y * z - w * x)
    rotation[2, 0] = wp.float64(2.0) * (x * z - w * y)
    rotation[2, 1] = wp.float64(2.0) * (y * z + w * x)
    rotation[2, 2] = wp.float64(1.0) - wp.float64(2.0) * (x * x + y * y)
    return rotation


@wp.func
def _quaternion_rotation_transpose_entry(
    base_pose: wp.array2d[wp.float64], environment: int, row: int, column: int
) -> wp.float64:
    """Return one normalized quaternion rotation-transpose entry.

    Args:
        base_pose: Batched spatial poses with shape ``(E, 7)`` in
            ``[qx, qy, qz, qw, x, y, z]`` order.
        environment: Environment row in ``base_pose``.
        row: Requested matrix row.
        column: Requested matrix column.

    Returns:
        Entry ``(row, column)`` of the normalized rotation transpose. A zero
        quaternion follows the trace-safe JAX path and maps to identity.
    """
    quaternion = _load_normalized_quaternion(base_pose, environment)
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    value = wp.float64(0.0)
    if row == 0 and column == 0:
        value = wp.float64(1.0) - wp.float64(2.0) * (y * y + z * z)
    elif row == 0 and column == 1:
        value = wp.float64(2.0) * (x * y + w * z)
    elif row == 0 and column == 2:
        value = wp.float64(2.0) * (x * z - w * y)
    elif row == 1 and column == 0:
        value = wp.float64(2.0) * (x * y - w * z)
    elif row == 1 and column == 1:
        value = wp.float64(1.0) - wp.float64(2.0) * (x * x + z * z)
    elif row == 1 and column == 2:
        value = wp.float64(2.0) * (y * z + w * x)
    elif row == 2 and column == 0:
        value = wp.float64(2.0) * (x * z + w * y)
    elif row == 2 and column == 1:
        value = wp.float64(2.0) * (y * z - w * x)
    elif row == 2 and column == 2:
        value = wp.float64(1.0) - wp.float64(2.0) * (x * x + y * y)
    return value


__all__ = [
    "_quaternion_rotation_matrix",
    "_quaternion_rotation_transpose_entry",
]
