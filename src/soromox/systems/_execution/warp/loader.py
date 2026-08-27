"""Single lazy import boundary for optional Warp dynamics executors."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from importlib import import_module
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from soromox.systems._execution.transforms import make_dynamics_evaluator
from soromox.systems._execution.types import (
    DynamicsEvaluator,
    DynamicsModel,
    DynamicsTerms,
    WarpExecutorKey,
)
from soromox.systems._execution.warp.config import gvs_block_dim
from soromox.systems._execution.warp.gvs.operands import GVSOperands
from soromox.systems._execution.warp.pcs.operands import PCSOperands

WarpExecutor = Callable[[Any, Array, Array], DynamicsTerms]

_EXECUTOR_MODULES: dict[WarpExecutorKey, str] = {
    "gvs": "soromox.systems._execution.warp.gvs.executor",
    "pcs": "soromox.systems._execution.warp.pcs.executor",
}


@cache
def load_executor(key: WarpExecutorKey) -> WarpExecutor:
    """Load one family executor without importing Warp during package import."""

    module_name = _EXECUTOR_MODULES[key]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == "warp":
            raise ImportError(
                "The requested dynamics executor requires the optional "
                "'warp-lang' dependency. Install it with "
                "`pip install soromox[warp]`."
            ) from error
        raise
    return cast(WarpExecutor, module.execute_dynamics_terms)


def execute_dynamics_terms(
    key: WarpExecutorKey,
    operands: Any,
    q: Array,
    qd: Array,
) -> DynamicsTerms:
    """Invoke the lazily resolved family executor."""

    return load_executor(key)(operands, q, qd)


def _validate_batch(q: Array, qd: Array, family_name: str) -> None:
    if q.shape[0] == 0:
        raise ValueError("The Warp executor requires a non-empty batch.")
    if q.dtype != jnp.float64 or qd.dtype != jnp.float64:
        raise TypeError(
            f"The Warp {family_name} dynamics executor requires float64 q "
            f"and qd; got {q.dtype} and {qd.dtype}."
        )


@eqx.filter_jit
def _execute_gvs_batch(
    model: DynamicsModel, q: Array, qd: Array
) -> DynamicsTerms:
    """Build GVS operands and execute one batch-shaped Warp pipeline."""

    _validate_batch(q, qd, "GVS")
    block_dim = gvs_block_dim(
        model.num_dofs,
        gpu=jax.default_backend() == "gpu",
    )
    operands = GVSOperands.from_model(model, block_dim=block_dim)
    return execute_dynamics_terms("gvs", operands, q, qd)


@eqx.filter_jit
def _execute_pcs_batch(
    model: DynamicsModel, q: Array, qd: Array
) -> DynamicsTerms:
    """Build PCS operands and execute one batch-shaped Warp pipeline."""

    _validate_batch(q, qd, "PCS")
    return execute_dynamics_terms("pcs", PCSOperands.from_model(model), q, qd)


def _call_gvs_batch(
    model: DynamicsModel, q: Array, qd: Array
) -> DynamicsTerms:
    """Resolve the GVS executor dynamically for tests and profiling."""

    return _execute_gvs_batch(model, q, qd)


def _call_pcs_batch(
    model: DynamicsModel, q: Array, qd: Array
) -> DynamicsTerms:
    """Resolve the PCS executor dynamically for tests and profiling."""

    return _execute_pcs_batch(model, q, qd)


_GVS_EVALUATOR = make_dynamics_evaluator(_call_gvs_batch, family_name="GVS")
_PCS_EVALUATOR = make_dynamics_evaluator(_call_pcs_batch, family_name="PCS")


def get_dynamics_evaluator(key: WarpExecutorKey) -> DynamicsEvaluator:
    """Return scalar semantics around a family batch executor."""

    if key == "gvs":
        return _GVS_EVALUATOR
    return _PCS_EVALUATOR


__all__ = [
    "WarpExecutor",
    "WarpExecutorKey",
    "execute_dynamics_terms",
    "get_dynamics_evaluator",
    "load_executor",
]
