"""Shared types and structural contracts for system execution backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from jax import Array

ExecutionBackend = Literal["auto", "jax", "warp"]
WarpExecutorKey = Literal["gvs", "pcs"]
DynamicsTerms = tuple[Array, Array, Array]


class DynamicsModel(Protocol):
    """Structural interface required by dynamics execution policy.

    System implementations satisfy this protocol implicitly; they do not
    inherit from an execution-layer base class. This keeps the neutral dispatch
    package independent of concrete model modules and prevents circular
    imports.

    Attributes:
        backend: Model-level execution preference used when a call does not
            supply an override.
        num_dofs: Number of active generalized coordinates in the model.
    """

    backend: ExecutionBackend
    num_dofs: int

    def _assemble_dynamics_terms(self, q: Array, qd: Array) -> DynamicsTerms: ...


class ForwardDynamicsModel(DynamicsModel, Protocol):
    """Model contract required by transform-aware forward dynamics.

    Implementations provide a compiled forward calculation whose ``backend``
    argument controls only dynamics-term assembly. The shared transformation
    boundary invokes this method with the configured backend for primal calls
    and with JAX for derivative calls.

    The protocol deliberately extends :class:`DynamicsModel` rather than
    importing ``GVS``, ``PCS``, or ``PlanarPCS``. Every supported system can
    therefore share the same transformation rule without creating a dependency
    from the execution layer back to the system classes.
    """

    def _evaluate_forward_dynamics(
        self,
        t: Array,
        y: Array,
        actuation_args: tuple | None,
        *,
        backend: ExecutionBackend | None,
    ) -> Array: ...


class DynamicsEvaluator(Protocol):
    """Callable contract for a transform-aware dynamics evaluator.

    Evaluators present scalar inputs to model code while retaining an internal
    batch-shaped executor. Their custom batching rule combines a mapped leading
    dimension into one executor call, and their derivative rule substitutes
    the model's differentiable JAX assembly.
    """

    def __call__(self, model: DynamicsModel, q: Array, qd: Array) -> DynamicsTerms: ...


@dataclass(frozen=True)
class DynamicsCapabilities:
    """Describe optional dynamics support for one system family.

    Attributes:
        family_name: Human-readable family name used in validation errors.
        warp_executor: Lazy-loader key for the family's Warp implementation.
        warp_cpu_supported: Whether the Warp executor may run when JAX's default
            backend is CPU. When false, automatic and explicit selection fall
            back to JAX on CPU.
        required_num_gauss_points: Exact quadrature size required by Warp, or
            ``None`` when the executor is shape-generic in quadrature count.
    """

    family_name: str
    warp_executor: WarpExecutorKey
    warp_cpu_supported: bool = False
    required_num_gauss_points: int | None = None


__all__ = [
    "DynamicsCapabilities",
    "DynamicsEvaluator",
    "DynamicsModel",
    "DynamicsTerms",
    "ExecutionBackend",
    "ForwardDynamicsModel",
    "WarpExecutorKey",
]
