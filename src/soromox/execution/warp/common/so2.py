# ruff: noqa: I001, UP018
"""Shared Warp operators for planar rotations."""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.func
def _rotation_transpose_entry(theta: wp.float64, row: int, column: int) -> wp.float64:
    """Return one entry of the transpose of a planar rotation.

    Args:
        theta: Rotation angle.
        row: Requested matrix row.
        column: Requested matrix column.

    Returns:
        Entry ``(row, column)`` of ``R(theta).T``.
    """
    cosine = wp.cos(theta)
    sine = wp.sin(theta)
    value = wp.float64(0.0)
    if row == 0 and column == 0 or row == 1 and column == 1:
        value = cosine
    elif row == 0 and column == 1:
        value = sine
    elif row == 1 and column == 0:
        value = -sine
    return value


__all__ = ["_rotation_transpose_entry"]
