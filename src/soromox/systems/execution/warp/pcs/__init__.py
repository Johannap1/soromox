"""Public Warp-native PlanarPCS and PCS dynamics building blocks.

The package exposes a shared runtime operand and workspace-shape contract plus
dimension-specific raw kernels and direct launch functions. All launch
functions accept caller-owned ``warp.array`` inputs and outputs; they do not
allocate buffers or invoke JAX, which allows an external integrator to compose
them into a larger CUDA graph.

Ordinary Soromox users should call the system methods. This lower-level surface
is intended for integration packages that need PlanarPCS or PCS mechanics
inside another Warp-native simulation loop.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "PCSOperandSource",
    "PCSOperands",
    "PCSPipelineShapes",
    "launch_planar_local_operators",
    "launch_planar_persistent_chain",
    "launch_spatial_local_operators",
    "launch_spatial_persistent_chain",
    "planar_local_operators_kernel",
    "planar_persistent_chain_kernel",
    "spatial_local_operators_kernel",
    "spatial_persistent_chain_kernel",
]

_PUBLIC_MODULES = {
    "PCSOperandSource": "operands",
    "PCSOperands": "operands",
    "PCSPipelineShapes": "operands",
    "launch_planar_local_operators": "planar_kernels",
    "launch_planar_persistent_chain": "planar_kernels",
    "launch_spatial_local_operators": "spatial_kernels",
    "launch_spatial_persistent_chain": "spatial_kernels",
    "planar_local_operators_kernel": "planar_kernels",
    "planar_persistent_chain_kernel": "planar_kernels",
    "spatial_local_operators_kernel": "spatial_kernels",
    "spatial_persistent_chain_kernel": "spatial_kernels",
}


def __getattr__(name: str) -> Any:
    """Load a public PCS Warp symbol without making Warp a core dependency.

    Args:
        name: Attribute requested from this package.

    Returns:
        The public operand type, kernel, or launch function named by ``name``.

    Raises:
        AttributeError: If ``name`` is not part of the public PCS Warp API.
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
    """Include lazily exported Warp symbols in interactive discovery.

    Returns:
        Sorted module globals and public lazy exports.
    """

    return sorted((*globals(), *__all__))
