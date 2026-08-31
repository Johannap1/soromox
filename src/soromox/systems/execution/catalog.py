"""Execution capabilities of production continuum-system families."""

from soromox.systems.execution.types import DynamicsCapabilities, KinematicsCapabilities

GVS_DYNAMICS = DynamicsCapabilities(
    family_name="GVS",
    warp_executor="gvs",
    warp_cpu_supported=True,
)
PCS_DYNAMICS = DynamicsCapabilities(
    family_name="PCS",
    warp_executor="pcs",
    required_num_gauss_points=5,
)
GVS_KINEMATICS = KinematicsCapabilities(
    family_name="GVS",
    warp_executor="gvs",
)
PCS_KINEMATICS = KinematicsCapabilities(
    family_name="PCS",
    warp_executor="pcs",
)

__all__ = [
    "GVS_DYNAMICS",
    "GVS_KINEMATICS",
    "PCS_DYNAMICS",
    "PCS_KINEMATICS",
]
