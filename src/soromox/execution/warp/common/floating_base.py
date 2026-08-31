# ruff: noqa: I001, UP018
"""Shared floating-base root operators for continuum Warp kernels."""

from __future__ import annotations

import warp as wp

from soromox.execution.warp.common.rotations import (
    _quaternion_rotation_transpose_entry,
)

wp.set_module_options({"enable_backward": False})


@wp.func
def _spatial_root_jacobian_entry(
    base_pose: wp.array2d[wp.float64], environment: int, row: int, column: int
) -> wp.float64:
    """Return one root body-Jacobian entry for world-frame base velocity.

    Args:
        base_pose: Batched scalar-last quaternion poses with shape ``(E, 7)``.
        environment: Environment row in ``base_pose``.
        row: Requested spatial row.
        column: Requested base-velocity column.

    Returns:
        Entry of ``diag(R.T, R.T)`` in angular-first spatial ordering.
    """
    if row < 3 and column < 3:
        return _quaternion_rotation_transpose_entry(base_pose, environment, row, column)
    if row >= 3 and column >= 3:
        return _quaternion_rotation_transpose_entry(
            base_pose, environment, row - 3, column - 3
        )
    return wp.float64(0.0)


@wp.kernel(enable_backward=False)
def _prepend_zeros_kernel(
    internal: wp.array2d[wp.float64],
    prefix: int,
    output: wp.array2d[wp.float64],
):
    """Copy an internal generalized vector behind an exact zero prefix."""
    environment, column = wp.tid()
    value = wp.float64(0.0)
    if column >= prefix:
        value = internal[environment, column - prefix]
    output[environment, column] = value


def launch_prepend_zeros(
    internal: wp.array2d[wp.float64],
    prefix: int,
    output: wp.array2d[wp.float64],
):
    """Launch a specialized zero-prefix copy into caller-owned storage.

    Args:
        internal: Batched internal generalized vectors.
        prefix: Number of leading floating-base entries.
        output: Batched total generalized-vector output.

    Returns:
        None. ``output`` is updated in place.
    """
    wp.launch(
        _prepend_zeros_kernel,
        dim=output.shape,
        inputs=[internal, prefix],
        outputs=[output],
    )


__all__ = [
    "_spatial_root_jacobian_entry",
    "launch_prepend_zeros",
]
