"""Public Warp-native GVS dynamics building blocks.

The kernels and launch functions in this module operate entirely on
``warp.array`` objects. They do not require JAX and do not allocate output
buffers, making them suitable for external CUDA-graph-capturable integrators
such as a Newton solver. Callers own the buffers and must preserve the shapes,
dtypes, coordinate maps, and launch ordering described by each function.

Soromox system users should normally call :meth:`GVS.dynamics_terms
<soromox.systems.GVS.dynamics_terms>` or :meth:`GVS.forward_dynamics
<soromox.systems.GVS.forward_dynamics>` instead. The lower-level API is intended
for integration packages that need to compose the mechanics pipeline inside a
larger Warp-native simulation loop.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "GVSOperandSource",
    "GVSOperands",
    "GVSPipelineShapes",
    "cell_terms_kernel",
    "cooperative_joint_terms_kernel",
    "joint_terms_kernel",
    "launch_cell_terms",
    "launch_cooperative_joint_terms",
    "launch_joint_terms",
    "launch_persistent_chain",
    "persistent_chain_kernel",
]

_PUBLIC_MODULES = {
    "GVSOperandSource": "operands",
    "GVSOperands": "operands",
    "GVSPipelineShapes": "operands",
    "cell_terms_kernel": "cell",
    "cooperative_joint_terms_kernel": "joint_cooperative",
    "joint_terms_kernel": "joint",
    "launch_cell_terms": "cell",
    "launch_cooperative_joint_terms": "joint_cooperative",
    "launch_joint_terms": "joint",
    "launch_persistent_chain": "chain",
    "persistent_chain_kernel": "chain",
}


def __getattr__(name: str) -> Any:
    """Load a public GVS Warp symbol without making Warp a core dependency.

    Args:
        name: Attribute requested from this package.

    Returns:
        The public operand type, kernel, or launch function named by ``name``.

    Raises:
        AttributeError: If ``name`` is not part of the public GVS Warp API.
        ImportError: If a kernel is requested without ``warp-lang`` installed.
    """

    try:
        module_name = _PUBLIC_MODULES[name]
    except KeyError as error:
        raise AttributeError(name) from error
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily exported Warp symbols in interactive discovery."""

    return sorted((*globals(), *__all__))
