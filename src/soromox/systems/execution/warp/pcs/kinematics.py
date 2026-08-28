"""Public dimension-selecting launch helpers for PCS kinematics."""

from __future__ import annotations

from typing import Any

from soromox.systems.execution.warp.pcs.planar_kinematics import (
    launch_planar_forward_kinematics,
    launch_planar_inertial_jacobians,
    launch_planar_kinematics,
)
from soromox.systems.execution.warp.pcs.spatial_kinematics import (
    launch_spatial_forward_kinematics,
    launch_spatial_inertial_jacobians,
    launch_spatial_kinematics,
)


def _selected_launcher(is_planar: bool, operation: str):
    """Return the public dimension-specialized PCS launch function.

    Args:
        is_planar: Select PlanarPCS when true and spatial PCS otherwise.
        operation: One of ``"pose"``, ``"jacobian"``, or ``"both"``.

    Returns:
        The matching public dimension- and operation-specialized launcher.
    """

    if operation == "pose":
        return (
            launch_planar_forward_kinematics
            if is_planar
            else launch_spatial_forward_kinematics
        )
    if operation == "jacobian":
        return (
            launch_planar_inertial_jacobians
            if is_planar
            else launch_spatial_inertial_jacobians
        )
    return launch_planar_kinematics if is_planar else launch_spatial_kinematics


def launch_forward_kinematics(*args: Any, is_planar: bool, **kwargs: Any) -> None:
    """Launch PCS forward kinematics with caller-owned Warp arrays.

    Args:
        *args: Dimension-specific positional arguments documented by
            ``launch_planar_kinematics`` and ``launch_spatial_kinematics``.
        is_planar: Select the PlanarPCS or spatial PCS kernel family.
        **kwargs: Dimension-specific keyword arguments forwarded unchanged.

    Returns:
        None. Outputs are written in place.
    """

    _selected_launcher(is_planar, "pose")(*args, **kwargs)


def launch_inertial_jacobians(*args: Any, is_planar: bool, **kwargs: Any) -> None:
    """Launch PCS inertial Jacobians through the shared fused recurrence.

    Args:
        *args: Dimension-specific positional arguments documented by the
            public specialized launcher.
        is_planar: Select the PlanarPCS or spatial PCS kernel family.
        **kwargs: Dimension-specific keyword arguments forwarded unchanged.

    Returns:
        None. Outputs are written in place.
    """

    _selected_launcher(is_planar, "jacobian")(*args, **kwargs)


def launch_forward_kinematics_and_jacobians(
    *args: Any, is_planar: bool, **kwargs: Any
) -> None:
    """Launch fused PCS poses and inertial Jacobians.

    Args:
        *args: Dimension-specific positional arguments documented by the
            public specialized launcher.
        is_planar: Select the PlanarPCS or spatial PCS kernel family.
        **kwargs: Dimension-specific keyword arguments forwarded unchanged.

    Returns:
        None. Outputs are written in place.
    """

    _selected_launcher(is_planar, "both")(*args, **kwargs)


__all__ = [
    "launch_forward_kinematics",
    "launch_forward_kinematics_and_jacobians",
    "launch_inertial_jacobians",
]
