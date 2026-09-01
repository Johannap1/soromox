# ruff: noqa: I001, UP018
"""Shared Warp operators for planar rotations."""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.func
def _rotation_components(theta: wp.float64) -> wp.vec2d:
    """Return the cosine and sine shared by planar rotation actions.

    Args:
        theta: Right-handed planar rotation angle.

    Returns:
        Vector ``[cos(theta), sin(theta)]``.
    """
    return wp.vec2d(wp.cos(theta), wp.sin(theta))


@wp.func
def _rotation_matrix(theta: wp.float64) -> wp.mat22d:
    """Return the planar rotation for ``theta``.

    Args:
        theta: Right-handed planar rotation angle.

    Returns:
        Two-dimensional active rotation matrix.
    """
    components = _rotation_components(theta)
    cosine = components[0]
    sine = components[1]
    return wp.mat22d(cosine, -sine, sine, cosine)


@wp.func
def _rotate_vector(theta: wp.float64, value: wp.vec2d) -> wp.vec2d:
    """Apply a planar active rotation to one vector.

    Args:
        theta: Right-handed planar rotation angle.
        value: Two-dimensional vector to rotate.

    Returns:
        Rotated vector ``R(theta) @ value``.
    """
    components = _rotation_components(theta)
    cosine = components[0]
    sine = components[1]
    return wp.vec2d(
        cosine * value[0] - sine * value[1],
        sine * value[0] + cosine * value[1],
    )


@wp.func
def _rotate_vector_transpose(theta: wp.float64, value: wp.vec2d) -> wp.vec2d:
    """Apply the transpose of a planar rotation to one vector.

    Args:
        theta: Right-handed planar rotation angle.
        value: Two-dimensional vector to rotate.

    Returns:
        Rotated vector ``R(theta).T @ value``.
    """
    components = _rotation_components(theta)
    cosine = components[0]
    sine = components[1]
    return wp.vec2d(
        cosine * value[0] + sine * value[1],
        -sine * value[0] + cosine * value[1],
    )


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
    components = _rotation_components(theta)
    cosine = components[0]
    sine = components[1]
    value = wp.float64(0.0)
    if row == 0 and column == 0 or row == 1 and column == 1:
        value = cosine
    elif row == 0 and column == 1:
        value = sine
    elif row == 1 and column == 0:
        value = -sine
    return value


__all__ = [
    "_rotate_vector",
    "_rotate_vector_transpose",
    "_rotation_components",
    "_rotation_matrix",
    "_rotation_transpose_entry",
]
