"""JAX transformation rules shared by optional dynamics executors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from soromox.systems.execution.types import (
    DynamicsEvaluator,
    DynamicsModel,
    DynamicsTerms,
    ForwardDynamicsModel,
)

BatchExecutor = Callable[[DynamicsModel, Array, Array], DynamicsTerms]


def make_dynamics_evaluator(
    execute_batch: BatchExecutor, *, family_name: str
) -> DynamicsEvaluator:
    """Adapt a batch-shaped executor to JAX scalar transformation semantics.

    The returned callable accepts one environment. Applying :func:`jax.vmap`
    does not replicate batch-one executor launches: the custom batching rule
    moves the mapped axis into a single call to ``execute_batch``. Applying
    forward- or reverse-mode differentiation replaces the forward-only executor
    with ``model._assemble_dynamics_terms`` and differentiates that JAX
    implementation. This is an execution-routing rule, not an alternative
    analytical dynamics derivative.

    Args:
        execute_batch: Family implementation accepting ``q`` and ``qd`` with
            shape ``(batch_size, num_dofs)`` and returning batched terms.
        family_name: Human-readable family name used in batching errors.

    Returns:
        A scalar-semantics dynamics evaluator with custom ``vmap`` and JVP
        behavior.
    """

    @jax.custom_batching.custom_vmap
    def execute_primal(model: DynamicsModel, q: Array, qd: Array) -> DynamicsTerms:
        """Execute a scalar request through a temporary one-item batch."""

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
        """Combine mapped environments into one family executor call.

        Args:
            axis_size: Size of the mapped axis.
            in_batched: Boolean PyTree indicating which inputs carry that axis.
            model: Unbatched system model.
            q: Scalar or mapped generalized coordinates.
            qd: Scalar or mapped generalized velocities.

        Returns:
            Batched dynamics terms and flags marking every output as batched.

        Raises:
            ValueError: If a caller attempts to batch model parameters.
        """

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
    def evaluate_terms(model: DynamicsModel, q: Array, qd: Array) -> DynamicsTerms:
        """Evaluate scalar forward-only dynamics terms."""

        return execute_primal(model, q, qd)

    @evaluate_terms.def_jvp
    def jvp_rule(
        primals: tuple[DynamicsModel, Array, Array],
        tangents: tuple[Any, Array | None, Array | None],
    ) -> tuple[DynamicsTerms, DynamicsTerms]:
        """Differentiate the model's JAX dynamics assembly.

        Args:
            primals: Model, generalized coordinates, and velocities.
            tangents: Equinox-filtered tangents corresponding to ``primals``.

        Returns:
            JAX-backed primal terms and their directional derivatives.
        """

        return eqx.filter_jvp(
            lambda model, q, qd: model._assemble_dynamics_terms(q, qd),
            primals,
            tangents,
        )

    return evaluate_terms


@eqx.filter_custom_jvp
def evaluate_forward_dynamics(
    model: ForwardDynamicsModel,
    t: Array,
    y: Array,
    actuation_args: tuple | None,
) -> Array:
    """Evaluate compiled forward dynamics with transform-aware backend routing.

    This boundary does not implement a new analytical derivative. Its custom
    JVP exists solely to prevent a compiled primal calculation from staging a
    forward-only Warp callback before JAX selects a derivative rule. Primal
    calls use the model's configured backend; forward- and reverse-mode
    transformations evaluate the same forward-dynamics equations with JAX term
    assembly.

    Args:
        model: System implementing the forward-dynamics execution contract.
        t: Current integration time.
        y: State vector accepted by the system's forward dynamics.
        actuation_args: Optional actuation and external-force arguments forwarded
            unchanged to the system.

    Returns:
        The state derivative produced by the model's compiled forward-dynamics
        implementation.
    """

    return model._evaluate_forward_dynamics(t, y, actuation_args, backend=None)


@evaluate_forward_dynamics.def_jvp
def _evaluate_forward_dynamics_jvp(
    primals: tuple[ForwardDynamicsModel, Array, Array, tuple | None],
    tangents: tuple[Any, Any, Any, Any],
) -> tuple[Array, Array]:
    """Differentiate forward dynamics using the model's JAX term assembly.

    Args:
        primals: Model, time, state, and optional actuation arguments supplied to
            :func:`evaluate_forward_dynamics`.
        tangents: Tangents corresponding to ``primals`` as filtered by Equinox.

    Returns:
        A pair containing the JAX-backed primal state derivative and its
        directional derivative.
    """

    return eqx.filter_jvp(
        lambda model, t, y, actuation_args: model._evaluate_forward_dynamics(
            t, y, actuation_args, backend="jax"
        ),
        primals,
        tangents,
    )


__all__ = [
    "BatchExecutor",
    "evaluate_forward_dynamics",
    "make_dynamics_evaluator",
]
