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

from soromox.systems.execution.transforms import make_dynamics_evaluator
from soromox.systems.execution.types import (
    DynamicsEvaluator,
    DynamicsModel,
    DynamicsTerms,
    WarpExecutorKey,
)
from soromox.systems.execution.warp.config import gvs_block_dim
from soromox.systems.execution.warp.gvs.operands import GVSOperands
from soromox.systems.execution.warp.pcs.operands import PCSOperands

WarpExecutor = Callable[[Any, Array, Array], DynamicsTerms]

_EXECUTOR_MODULES: dict[WarpExecutorKey, str] = {
    "gvs": "soromox.systems.execution.warp.gvs.executor",
    "pcs": "soromox.systems.execution.warp.pcs.executor",
}


@cache
def load_executor(key: WarpExecutorKey) -> WarpExecutor:
    """Load and cache one family executor behind the optional-dependency boundary.

    Args:
        key: Registered system-family executor key.

    Returns:
        The family's batch-shaped ``execute_dynamics_terms`` function.

    Raises:
        KeyError: If ``key`` is not registered.
        ImportError: If the executor needs ``warp-lang`` and it is not installed.
        ModuleNotFoundError: If loading fails because another module is missing;
            unrelated import failures are deliberately preserved.
    """

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
    """Invoke a lazily resolved family executor with prepared operands.

    Args:
        key: Registered family executor key.
        operands: Family-specific immutable runtime operand bundle.
        q: Batched FP64 generalized coordinates.
        qd: Batched FP64 generalized velocities.

    Returns:
        Batched inertia, convective-force, and gravity-force terms.

    Raises:
        ImportError: If the optional Warp dependency is unavailable.
    """

    return load_executor(key)(operands, q, qd)


def _validate_batch(q: Array, qd: Array, family_name: str) -> None:
    """Validate restrictions shared by the production Warp executors.

    Args:
        q: Batched generalized coordinates.
        qd: Batched generalized velocities.
        family_name: Name included in dtype diagnostics.

    Raises:
        ValueError: If the leading batch dimension is empty.
        TypeError: If either input is not FP64.
    """

    if q.shape[0] == 0:
        raise ValueError("The Warp executor requires a non-empty batch.")
    if q.dtype != jnp.float64 or qd.dtype != jnp.float64:
        raise TypeError(
            f"The Warp {family_name} dynamics executor requires float64 q "
            f"and qd; got {q.dtype} and {qd.dtype}."
        )


@eqx.filter_jit
def _execute_gvs_batch(model: DynamicsModel, q: Array, qd: Array) -> DynamicsTerms:
    """Build GVS operands and execute one compiled batch-shaped pipeline.

    Args:
        model: GVS-compatible dynamics model.
        q: FP64 coordinates with shape ``(batch_size, num_dofs)``.
        qd: FP64 velocities with the same shape as ``q``.

    Returns:
        Batched ``(B, Cqd, G)`` terms.
    """

    _validate_batch(q, qd, "GVS")
    block_dim = gvs_block_dim(
        model.num_dofs,
        gpu=jax.default_backend() == "gpu",
    )
    operands = GVSOperands.from_model(model, block_dim=block_dim)
    return execute_dynamics_terms("gvs", operands, q, qd)


@eqx.filter_jit
def _execute_pcs_batch(model: DynamicsModel, q: Array, qd: Array) -> DynamicsTerms:
    """Build PCS operands and execute one compiled batch-shaped pipeline.

    Args:
        model: PlanarPCS- or PCS-compatible dynamics model.
        q: FP64 coordinates with shape ``(batch_size, num_dofs)``.
        qd: FP64 velocities with the same shape as ``q``.

    Returns:
        Batched ``(B, Cqd, G)`` terms.
    """

    _validate_batch(q, qd, "PCS")
    return execute_dynamics_terms("pcs", PCSOperands.from_model(model), q, qd)


def _call_gvs_batch(model: DynamicsModel, q: Array, qd: Array) -> DynamicsTerms:
    """Call the replaceable GVS batch boundary.

    Args:
        model: GVS-compatible dynamics model.
        q: Batched generalized coordinates.
        qd: Batched generalized velocities.

    Returns:
        Batched ``(B, Cqd, G)`` terms.

    Notes:
        Keeping this one-line boundary outside the evaluator closure permits
        focused instrumentation without coupling model code to Warp modules.
    """

    return _execute_gvs_batch(model, q, qd)


def _call_pcs_batch(model: DynamicsModel, q: Array, qd: Array) -> DynamicsTerms:
    """Call the replaceable PCS batch boundary.

    Args:
        model: PlanarPCS- or PCS-compatible dynamics model.
        q: Batched generalized coordinates.
        qd: Batched generalized velocities.

    Returns:
        Batched ``(B, Cqd, G)`` terms.

    Notes:
        Keeping this one-line boundary outside the evaluator closure permits
        focused instrumentation without coupling model code to Warp modules.
    """

    return _execute_pcs_batch(model, q, qd)


_GVS_EVALUATOR = make_dynamics_evaluator(_call_gvs_batch, family_name="GVS")
_PCS_EVALUATOR = make_dynamics_evaluator(_call_pcs_batch, family_name="PCS")


def get_dynamics_evaluator(key: WarpExecutorKey) -> DynamicsEvaluator:
    """Return the transform-aware evaluator for a registered family.

    Args:
        key: Registered family executor key.

    Returns:
        Scalar-semantics evaluator whose custom ``vmap`` invokes one batched
        executor and whose derivative rule uses JAX.
    """

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
