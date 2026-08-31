"""Execution capabilities of production continuum-system families."""

from soromox.systems.execution.types import DynamicsCapabilities

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

__all__ = ["GVS_DYNAMICS", "PCS_DYNAMICS"]
