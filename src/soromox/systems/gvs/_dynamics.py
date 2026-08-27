"""Transform-aware execution of GVS dynamics terms."""

from __future__ import annotations

from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

DynamicsTerms = tuple[Array, Array, Array]


class DynamicsModel(Protocol):
    """Structural interface required by the backend dispatcher."""

    num_dofs: int

    def _assemble_dynamics_terms(self, q: Array, qd: Array) -> DynamicsTerms: ...


@eqx.filter_jit
def _execute_batch(model: DynamicsModel, q: Array, qd: Array) -> DynamicsTerms:
    """Execute one shape-generic Warp launch pipeline for a state batch."""

    if q.shape[0] == 0:
        raise ValueError("The Warp backend requires a non-empty batch.")
    if q.dtype != jnp.float64 or qd.dtype != jnp.float64:
        raise TypeError(
            "The Warp GVS backend currently requires float64 q and qd; "
            f"got {q.dtype} and {qd.dtype}."
        )

    try:
        from soromox.systems.gvs._warp.backend import dynamics_terms
    except ModuleNotFoundError as error:
        if error.name == "warp":
            raise ImportError(
                "The Warp GVS backend requires the optional 'warp-lang' "
                "dependency. Install it with `pip install soromox[warp]`."
            ) from error
        raise

    if jax.default_backend() == "gpu":
        lanes_per_block = 192 if model.num_dofs > 64 else 128
    else:
        lanes_per_block = 1
    return dynamics_terms(model, q, qd, lanes_per_block=lanes_per_block)


@jax.custom_batching.custom_vmap
def _execute_primal(model: DynamicsModel, q: Array, qd: Array) -> DynamicsTerms:
    """Give the batched Warp implementation scalar-call semantics."""

    batched = _execute_batch(model, q[None, :], qd[None, :])
    return jax.tree.map(lambda value: value[0], batched)


@_execute_primal.def_vmap
def _vmap_rule(
    axis_size: int,
    in_batched: tuple[Any, bool, bool],
    model: DynamicsModel,
    q: Array,
    qd: Array,
) -> tuple[DynamicsTerms, tuple[bool, bool, bool]]:
    """Map independent environments to one batched Warp execution."""

    model_batched, q_batched, qd_batched = in_batched
    if any(jax.tree.leaves(model_batched)):
        raise ValueError("Batching over GVS model parameters is not supported.")
    if not q_batched:
        q = jnp.broadcast_to(q, (axis_size, *q.shape))
    if not qd_batched:
        qd = jnp.broadcast_to(qd, (axis_size, *qd.shape))
    return _execute_batch(model, q, qd), (True, True, True)


@eqx.filter_custom_jvp
def evaluate_terms(model: DynamicsModel, q: Array, qd: Array) -> DynamicsTerms:
    """Evaluate Warp dynamics while retaining differentiable JAX semantics."""

    return _execute_primal(model, q, qd)


@evaluate_terms.def_jvp
def _jvp_rule(
    primals: tuple[DynamicsModel, Array, Array],
    tangents: tuple[Any, Array | None, Array | None],
) -> tuple[DynamicsTerms, DynamicsTerms]:
    """Use the JAX assembly for both forward- and reverse-mode derivatives."""

    return eqx.filter_jvp(
        lambda model, q, qd: model._assemble_dynamics_terms(q, qd),
        primals,
        tangents,
    )
