"""Backend-independent execution policy for accelerated system operations."""

from soromox.systems._execution.catalog import GVS_DYNAMICS, PCS_DYNAMICS
from soromox.systems._execution.dispatch import dispatch_dynamics_terms
from soromox.systems._execution.transforms import make_dynamics_evaluator
from soromox.systems._execution.types import (
    DynamicsCapabilities,
    DynamicsEvaluator,
    DynamicsModel,
    DynamicsTerms,
    ExecutionBackend,
    WarpExecutorKey,
)
from soromox.systems._execution.warp.config import (
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
    "GVS_DYNAMICS",
    "PCS_DYNAMICS",
    "DEFAULT_PCS_BLOCK_DIM",
    "DEFAULT_PLANAR_PCS_BLOCK_DIM",
    "WarpExecutorKey",
    "dispatch_dynamics_terms",
    "make_dynamics_evaluator",
    "validate_block_dim",
]
