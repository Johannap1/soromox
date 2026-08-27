"""Execution backends for accelerated system operations.

This public package defines backend selection and transformation semantics used
by supported Soromox system classes. Most users only need
:class:`ExecutionBackend` and the ``backend`` constructor argument on a system
model.

Integrations that need direct Warp execution can import stable family-specific
building blocks from :mod:`soromox.systems.execution.warp`. Importing this
module does not import the optional ``warp-lang`` dependency; Warp is loaded
only when an integration explicitly enters that namespace or selects a Warp
executor.
"""

from soromox.systems.execution.catalog import GVS_DYNAMICS, PCS_DYNAMICS
from soromox.systems.execution.dispatch import dispatch_dynamics_terms
from soromox.systems.execution.transforms import (
    evaluate_forward_dynamics,
    make_dynamics_evaluator,
)
from soromox.systems.execution.types import (
    DynamicsCapabilities,
    DynamicsEvaluator,
    DynamicsModel,
    DynamicsTerms,
    ExecutionBackend,
    ForwardDynamicsModel,
    WarpExecutorKey,
)
from soromox.systems.execution.warp.config import (
    DEFAULT_PCS_BLOCK_DIM,
    DEFAULT_PLANAR_PCS_BLOCK_DIM,
    validate_block_dim,
)

__all__ = [
    "DynamicsCapabilities",
    "DynamicsEvaluator",
    "DynamicsModel",
    "DynamicsTerms",
    "ExecutionBackend",
    "ForwardDynamicsModel",
    "GVS_DYNAMICS",
    "PCS_DYNAMICS",
    "DEFAULT_PCS_BLOCK_DIM",
    "DEFAULT_PLANAR_PCS_BLOCK_DIM",
    "WarpExecutorKey",
    "dispatch_dynamics_terms",
    "evaluate_forward_dynamics",
    "make_dynamics_evaluator",
    "validate_block_dim",
]
