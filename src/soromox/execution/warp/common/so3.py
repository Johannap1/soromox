# ruff: noqa: I001, UP018
"""Shared Warp operators for spatial rotations."""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.func
def _rotation_entry(
    omega: wp.vec3d,
    angle_sq: wp.float64,
    exponential_coefficients: wp.vec3d,
    row: int,
    column: int,
) -> wp.float64:
    """Evaluate one entry of an SO(3) exponential rotation matrix.

    Args:
        omega: Rotational exponential coordinate.
        angle_sq: Squared norm of ``omega``.
        exponential_coefficients: Stable ``(sinc, cosc, tanc)`` coefficients.
        row: Matrix row index.
        column: Matrix column index.

    Returns:
        The selected rotation-matrix entry.
    """
    identity_entry = wp.float64(0.0)
    if row == column:
        identity_entry = wp.float64(1.0)
    skew_entry = wp.float64(0.0)
    if row == 0 and column == 1:
        skew_entry = -omega[2]
    elif row == 0 and column == 2:
        skew_entry = omega[1]
    elif row == 1 and column == 0:
        skew_entry = omega[2]
    elif row == 1 and column == 2:
        skew_entry = -omega[0]
    elif row == 2 and column == 0:
        skew_entry = -omega[1]
    elif row == 2 and column == 1:
        skew_entry = omega[0]
    skew_square_entry = omega[row] * omega[column] - angle_sq * identity_entry
    return (
        identity_entry
        + exponential_coefficients[0] * skew_entry
        + exponential_coefficients[1] * skew_square_entry
    )


__all__ = ["_rotation_entry"]
