"""Backend-neutral validation and selection for dynamics execution."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from soromox.systems.execution.types import (
    DynamicsCapabilities,
    DynamicsModel,
    DynamicsTerms,
    ExecutionBackend,
)
from soromox.systems.execution.warp.loader import get_dynamics_evaluator


def _select_backend(
    model: DynamicsModel,
    requested: ExecutionBackend | None,
    capabilities: DynamicsCapabilities,
    *,
    warp_supported: bool,
) -> ExecutionBackend:
    """Resolve a requested backend against device and model capabilities.

    Args:
        model: System supplying the configured backend and static dimensions.
        requested: Optional per-call override. ``None`` uses ``model.backend``.
        capabilities: Static support declared for the system family.
        warp_supported: Whether this concrete model instance is implemented by
            the family's Warp executor.

    Returns:
        The concrete backend to execute, either ``"jax"`` or ``"warp"``.

    Raises:
        ValueError: If the requested or configured backend name is invalid.
        NotImplementedError: If Warp was requested explicitly for an unsupported
            quadrature rule or model instance.
    """

    configured = model.backend if requested is None else requested
    if configured not in ("auto", "jax", "warp"):
        raise ValueError(
            f"backend must be one of 'auto', 'jax', or 'warp', got {requested!r}."
        )

    selected: ExecutionBackend = configured
    if selected == "auto":
        selected = "warp" if jax.default_backend() == "gpu" else "jax"
    if (
        selected == "warp"
        and jax.default_backend() != "gpu"
        and not capabilities.warp_cpu_supported
    ):
        selected = "jax"
    if model.num_dofs == 0:
        selected = "jax"

    required_points = capabilities.required_num_gauss_points
    actual_points = getattr(model, "num_gauss_points", None)
    if (
        selected == "warp"
        and required_points is not None
        and actual_points != required_points
    ):
        if configured == "warp":
            raise NotImplementedError(
                f"The Warp {capabilities.family_name} dynamics executor "
                f"requires exactly {required_points} Gauss points."
            )
        selected = "jax"
    if selected == "warp" and not warp_supported:
        if configured == "warp":
            raise NotImplementedError(
                f"The Warp {capabilities.family_name} dynamics executor is "
                "not enabled for this system."
            )
        selected = "jax"
    return selected


def dispatch_dynamics_terms(
    model: DynamicsModel,
    q: Array,
    qd: Array,
    *,
    backend: ExecutionBackend | None,
    capabilities: DynamicsCapabilities,
    warp_supported: bool = True,
) -> DynamicsTerms:
    """Validate inputs and invoke one system's selected dynamics assembly.

    Scalar inputs retain scalar outputs. A single leading environment dimension
    is also accepted. JAX batches are evaluated with :func:`jax.vmap`; Warp
    batches pass through the evaluator's custom batching rule and therefore use
    one batch-shaped executor invocation. Derivative transformations applied to
    either form are handled by the evaluator and use ``_assemble_dynamics_terms``.

    Args:
        model: System implementing the neutral :class:`DynamicsModel` contract.
        q: Generalized coordinates with shape ``(num_dofs,)`` or
            ``(batch_size, num_dofs)``.
        qd: Generalized velocities with the same shape and dtype as ``q``.
        backend: Optional per-call backend override. ``None`` uses the model's
            configured backend.
        capabilities: Static support declared for the system family.
        warp_supported: Whether this concrete model instance can use the family
            Warp executor. This distinguishes an exact supported class from an
            arbitrary subclass with different equations.

    Returns:
        A tuple ``(B, Cqd, G)``. Each result has the same optional leading batch
        dimension as the inputs.

    Raises:
        ValueError: If either state has an invalid shape, their shapes differ,
            or the backend name is invalid.
        NotImplementedError: If Warp is explicitly requested for an unsupported
            quadrature rule or model instance.
        ImportError: If Warp is selected but the optional dependency is absent.
        TypeError: If the selected Warp executor does not support the state
            dtype.
    """

    q = jnp.asarray(q)
    qd = jnp.asarray(qd)
    if q.ndim not in (1, 2) or q.shape[-1:] != (model.num_dofs,):
        raise ValueError(
            "q must have shape (num_dofs,) or (batch_size, num_dofs); "
            f"expected (..., {model.num_dofs}), got {q.shape}."
        )
    if qd.shape != q.shape:
        raise ValueError(f"qd must have shape {q.shape}, got {qd.shape}.")

    selected = _select_backend(
        model,
        backend,
        capabilities,
        warp_supported=warp_supported,
    )
    if selected == "jax":
        if q.ndim == 1:
            return model._assemble_dynamics_terms(q, qd)
        return jax.vmap(model._assemble_dynamics_terms)(q, qd)
    warp_evaluator = get_dynamics_evaluator(capabilities.warp_executor)
    if q.ndim == 1:
        return warp_evaluator(model, q, qd)
    return jax.vmap(warp_evaluator, in_axes=(None, 0, 0))(model, q, qd)


__all__ = ["dispatch_dynamics_terms"]
