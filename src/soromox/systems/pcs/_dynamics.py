"""Transform-aware execution of optional PlanarPCS and PCS acceleration."""

from __future__ import annotations

from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from soromox.systems._dynamics import ExecutionBackend, make_dynamics_evaluator

DynamicsTerms = tuple[Array, Array, Array]
DEFAULT_PLANAR_PCS_WARP_BLOCK_DIM = 128
DEFAULT_PCS_WARP_BLOCK_DIM = 192


def validate_warp_block_dim(value: int) -> int:
    """Validate a CUDA thread-block dimension for the PCS Warp kernels."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "warp_block_dim must be an integer multiple of 32, "
            f"got {type(value).__name__}."
        )
    if value < 32 or value > 1024 or value % 32 != 0:
        raise ValueError(
            "warp_block_dim must be a multiple of 32 between 32 and 1024, "
            f"got {value}."
        )
    return value


class PCSModel(Protocol):
    """Runtime properties shared by planar and spatial PCS models."""

    num_dofs: int
    num_gauss_points: int
    is_planar: bool
    backend: ExecutionBackend
    warp_block_dim: int

    def _assemble_dynamics_terms(self, q: Array, qd: Array) -> DynamicsTerms: ...


@eqx.filter_jit
def _execute_batch(model: PCSModel, q: Array, qd: Array) -> DynamicsTerms:
    """Execute one batch-shaped PCS Warp pipeline."""

    if q.shape[0] == 0:
        raise ValueError("The Warp backend requires a non-empty batch.")
    if q.dtype != jnp.float64 or qd.dtype != jnp.float64:
        raise TypeError(
            "The Warp PCS backend currently requires float64 q and qd; "
            f"got {q.dtype} and {qd.dtype}."
        )

    try:
        from soromox.systems.pcs._warp.backend import dynamics_terms
    except ModuleNotFoundError as error:
        if error.name == "warp":
            raise ImportError(
                "The Warp PCS backend requires the optional 'warp-lang' "
                "dependency. Install it with `pip install soromox[warp]`."
            ) from error
        raise

    return dynamics_terms(model, q, qd, block_dim=model.warp_block_dim)


def _call_execute_batch(
    model: PCSModel, q: Array, qd: Array
) -> DynamicsTerms:
    """Resolve the batch executor dynamically for tests and profiling."""

    return _execute_batch(model, q, qd)


evaluate_terms = make_dynamics_evaluator(_call_execute_batch, family_name="PCS")


def dispatch_terms(
    model: PCSModel,
    q: Array,
    qd: Array,
    *,
    backend: ExecutionBackend | None,
    warp_supported: bool,
) -> DynamicsTerms:
    """Validate shapes and select JAX or the transform-aware Warp evaluator."""

    q = jnp.asarray(q)
    qd = jnp.asarray(qd)
    if q.ndim not in (1, 2) or q.shape[-1:] != (model.num_dofs,):
        raise ValueError(
            "q must have shape (num_dofs,) or (batch_size, num_dofs); "
            f"expected (..., {model.num_dofs}), got {q.shape}."
        )
    if qd.shape != q.shape:
        raise ValueError(f"qd must have shape {q.shape}, got {qd.shape}.")

    configured_backend = model.backend if backend is None else backend
    if configured_backend not in ("auto", "jax", "warp"):
        raise ValueError(
            "backend must be one of 'auto', 'jax', or 'warp', "
            f"got {backend!r}."
        )
    selected_backend = configured_backend
    if selected_backend == "auto":
        selected_backend = "warp" if jax.default_backend() == "gpu" else "jax"
    if selected_backend == "warp" and jax.default_backend() != "gpu":
        selected_backend = "jax"
    if model.num_dofs == 0:
        selected_backend = "jax"
    if selected_backend == "warp" and model.num_gauss_points != 5:
        if configured_backend == "warp":
            raise NotImplementedError(
                "The Warp PCS dynamics backend currently requires exactly "
                "five Gauss points."
            )
        selected_backend = "jax"
    if selected_backend == "warp" and not warp_supported:
        if configured_backend == "warp":
            raise NotImplementedError(
                "The Warp PCS dynamics backend is not enabled for this subclass."
            )
        selected_backend = "jax"

    if selected_backend == "jax":
        if q.ndim == 1:
            return model._assemble_dynamics_terms(q, qd)
        return jax.vmap(model._assemble_dynamics_terms)(q, qd)
    if q.ndim == 1:
        return evaluate_terms(model, q, qd)
    return jax.vmap(evaluate_terms, in_axes=(None, 0, 0))(model, q, qd)


__all__ = ["dispatch_terms", "evaluate_terms"]
