"""Unit tests for backend selection and public dynamics dispatch."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array
from numpy.testing import assert_allclose

from soromox.systems.execution import (
    GVS_DYNAMICS,
    DynamicsCapabilities,
    ExecutionBackend,
    dispatch_dynamics_terms,
)

jax.config.update("jax_enable_x64", True)


class _DispatchProbe(eqx.Module):
    """Minimal public dynamics model used to observe dispatch behavior."""

    scale: Array
    backend: ExecutionBackend = eqx.field(static=True)
    num_dofs: int = eqx.field(static=True)
    num_gauss_points: int = eqx.field(static=True, default=5)

    def _assemble_dynamics_terms(
        self, q: Array, qd: Array
    ) -> tuple[Array, Array, Array]:
        """Return differentiable terms with the production output shapes."""

        inertia = self.scale * jnp.eye(self.num_dofs, dtype=q.dtype) + jnp.outer(q, q)
        return inertia, self.scale * qd, q + self.scale

    def dynamics_terms(
        self,
        q: Array,
        qd: Array,
        *,
        backend: ExecutionBackend | None = None,
    ) -> tuple[Array, Array, Array]:
        """Dispatch through the same public boundary as supported systems."""

        return dispatch_dynamics_terms(
            self,
            q,
            qd,
            backend=backend,
            capabilities=GVS_DYNAMICS,
        )


def _probe(
    *, backend: ExecutionBackend = "jax", gauss_points: int = 5
) -> _DispatchProbe:
    """Construct a three-DOF dispatch probe."""

    return _DispatchProbe(
        scale=jnp.asarray(2.0, dtype=jnp.float64),
        backend=backend,
        num_dofs=3,
        num_gauss_points=gauss_points,
    )


@pytest.mark.parametrize(
    ("q_shape", "qd_shape", "message"),
    [
        ((2,), (2,), "q must have shape"),
        ((1, 2, 3), (1, 2, 3), "q must have shape"),
        ((3,), (1, 3), "qd must have shape"),
    ],
)
def test_dispatch_validates_state_shapes(
    q_shape: tuple[int, ...],
    qd_shape: tuple[int, ...],
    message: str,
) -> None:
    """Reject states that do not follow the scalar-or-one-batch contract."""

    model = _probe()
    with pytest.raises(ValueError, match=message):
        model.dynamics_terms(jnp.zeros(q_shape), jnp.zeros(qd_shape))


def test_dispatch_rejects_unknown_backend() -> None:
    """Validate runtime strings even though the public type is a Literal."""

    model = _probe()
    state = jnp.zeros((model.num_dofs,), dtype=jnp.float64)
    with pytest.raises(ValueError, match="backend must be one of"):
        model.dynamics_terms(state, state, backend="unknown")  # type: ignore[arg-type]


def test_direct_vmap_reaches_one_batched_warp_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map the public scalar API without launching one Warp batch per item."""

    from soromox.systems.execution.warp import loader

    model = _probe(backend="warp")
    q = jnp.arange(12, dtype=jnp.float64).reshape(4, 3) / 10.0
    qd = -0.5 * q
    observed_batches: list[int] = []

    def fake_batch(
        model: _DispatchProbe, q: Array, qd: Array
    ) -> tuple[Array, Array, Array]:
        del qd
        batch_size = q.shape[0]
        jax.debug.callback(
            lambda value: observed_batches.append(int(value)),
            jnp.asarray(batch_size),
        )
        marker = jnp.asarray(batch_size, dtype=q.dtype)
        return (
            jnp.full((batch_size, model.num_dofs, model.num_dofs), marker),
            jnp.full((batch_size, model.num_dofs), marker + 1.0),
            jnp.full((batch_size, model.num_dofs), marker + 2.0),
        )

    monkeypatch.setattr(loader, "_execute_gvs_batch", fake_batch)
    monkeypatch.setattr(
        "soromox.systems.execution.dispatch.jax.default_backend",
        lambda: "gpu",
    )

    mapped = jax.vmap(model.dynamics_terms)(q, qd)

    assert observed_batches == [4]
    assert_allclose(mapped[0], jnp.full((4, 3, 3), 4.0), rtol=0.0, atol=0.0)


def test_explicit_batch_reaches_one_batched_warp_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve one executor call for the explicit leading-batch API."""

    from soromox.systems.execution.warp import loader

    model = _probe(backend="warp")
    q = jnp.ones((5, model.num_dofs), dtype=jnp.float64)
    observed_batches: list[int] = []

    def fake_batch(
        model: _DispatchProbe, q: Array, qd: Array
    ) -> tuple[Array, Array, Array]:
        del qd
        batch_size = q.shape[0]
        jax.debug.callback(
            lambda value: observed_batches.append(int(value)),
            jnp.asarray(batch_size),
        )
        return (
            jnp.zeros((batch_size, model.num_dofs, model.num_dofs)),
            jnp.zeros((batch_size, model.num_dofs)),
            jnp.zeros((batch_size, model.num_dofs)),
        )

    monkeypatch.setattr(loader, "_execute_gvs_batch", fake_batch)
    monkeypatch.setattr(
        "soromox.systems.execution.dispatch.jax.default_backend",
        lambda: "gpu",
    )

    model.dynamics_terms(q, q)

    assert observed_batches == [5]


def test_explicit_warp_reports_unsupported_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinguish an explicit unsupported request from an automatic fallback."""

    monkeypatch.setattr(
        "soromox.systems.execution.dispatch.jax.default_backend",
        lambda: "gpu",
    )
    model = _probe(backend="warp")
    state = jnp.zeros((model.num_dofs,), dtype=jnp.float64)

    with pytest.raises(NotImplementedError, match="not enabled"):
        dispatch_dynamics_terms(
            model,
            state,
            state,
            backend=None,
            capabilities=GVS_DYNAMICS,
            warp_supported=False,
        )


def test_auto_falls_back_for_unsupported_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep ``auto`` usable when a model instance cannot use Warp."""

    monkeypatch.setattr(
        "soromox.systems.execution.dispatch.jax.default_backend",
        lambda: "gpu",
    )
    model = _probe(backend="auto")
    state = jnp.linspace(-0.1, 0.1, model.num_dofs, dtype=jnp.float64)
    expected = model._assemble_dynamics_terms(state, state)
    actual = dispatch_dynamics_terms(
        model,
        state,
        state,
        backend=None,
        capabilities=GVS_DYNAMICS,
        warp_supported=False,
    )

    for actual_term, expected_term in zip(actual, expected, strict=True):
        assert_allclose(actual_term, expected_term)


def test_required_quadrature_is_checked_before_executor_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report the five-point PCS restriction without importing Warp."""

    monkeypatch.setattr(
        "soromox.systems.execution.dispatch.jax.default_backend",
        lambda: "gpu",
    )
    capabilities = DynamicsCapabilities(
        family_name="test PCS",
        warp_executor="pcs",
        required_num_gauss_points=5,
    )
    model = _probe(backend="warp", gauss_points=3)
    state = jnp.zeros((model.num_dofs,), dtype=jnp.float64)

    with pytest.raises(NotImplementedError, match="exactly 5 Gauss points"):
        dispatch_dynamics_terms(
            model,
            state,
            state,
            backend=None,
            capabilities=capabilities,
        )
