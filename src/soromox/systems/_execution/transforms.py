"""JAX transformation rules shared by optional dynamics executors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from soromox.systems._execution.types import (
    DynamicsEvaluator,
    DynamicsModel,
    DynamicsTerms,
)

BatchExecutor = Callable[[DynamicsModel, Array, Array], DynamicsTerms]


def make_dynamics_evaluator(
    execute_batch: BatchExecutor, *, family_name: str
) -> DynamicsEvaluator:
    """Give one batch-shaped primal executor scalar and derivative semantics."""

    @jax.custom_batching.custom_vmap
    def execute_primal(
        model: DynamicsModel, q: Array, qd: Array
    ) -> DynamicsTerms:
        batched = execute_batch(model, q[None, :], qd[None, :])
        return jax.tree.map(lambda value: value[0], batched)

    @execute_primal.def_vmap
    def vmap_rule(
        axis_size: int,
        in_batched: tuple[Any, bool, bool],
        model: DynamicsModel,
        q: Array,
        qd: Array,
    ) -> tuple[DynamicsTerms, tuple[bool, bool, bool]]:
        model_batched, q_batched, qd_batched = in_batched
        if any(jax.tree.leaves(model_batched)):
            raise ValueError(
                f"Batching over {family_name} model parameters is not supported."
            )
        if not q_batched:
            q = jnp.broadcast_to(q, (axis_size, *q.shape))
        if not qd_batched:
            qd = jnp.broadcast_to(qd, (axis_size, *qd.shape))
        return execute_batch(model, q, qd), (True, True, True)

    @eqx.filter_custom_jvp
    def evaluate_terms(
        model: DynamicsModel, q: Array, qd: Array
    ) -> DynamicsTerms:
        return execute_primal(model, q, qd)

    @evaluate_terms.def_jvp
    def jvp_rule(
        primals: tuple[DynamicsModel, Array, Array],
        tangents: tuple[Any, Array | None, Array | None],
    ) -> tuple[DynamicsTerms, DynamicsTerms]:
        return eqx.filter_jvp(
            lambda model, q, qd: model._assemble_dynamics_terms(q, qd),
            primals,
            tangents,
        )

    return evaluate_terms


__all__ = ["BatchExecutor", "make_dynamics_evaluator"]
