"""Transform-aware execution of GVS dynamics terms."""

from __future__ import annotations

from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from soromox.systems._dynamics import make_dynamics_evaluator

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


def _call_execute_batch(
    model: DynamicsModel, q: Array, qd: Array
) -> DynamicsTerms:
    """Resolve the executor dynamically so tests can replace it safely."""

    return _execute_batch(model, q, qd)


evaluate_terms = make_dynamics_evaluator(_call_execute_batch, family_name="GVS")
