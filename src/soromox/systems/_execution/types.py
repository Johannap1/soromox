"""Shared types and structural contracts for system execution backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from jax import Array

ExecutionBackend = Literal["auto", "jax", "warp"]
WarpExecutorKey = Literal["gvs", "pcs"]
DynamicsTerms = tuple[Array, Array, Array]


class DynamicsModel(Protocol):
    """Model interface required by dynamics dispatch and transformations."""

    backend: ExecutionBackend
    num_dofs: int

    def _assemble_dynamics_terms(self, q: Array, qd: Array) -> DynamicsTerms: ...


class DynamicsEvaluator(Protocol):
    """Scalar-semantics evaluator that may provide a custom batching rule."""

    def __call__(
        self, model: DynamicsModel, q: Array, qd: Array
    ) -> DynamicsTerms: ...


@dataclass(frozen=True)
class DynamicsCapabilities:
    """Static restrictions of one optional dynamics implementation."""

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
    "WarpExecutorKey",
]
