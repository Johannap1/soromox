"""JAX/Warp equivalence tests for five-point spatial PCS dynamics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import pytest

from soromox.systems.execution.warp.pcs.operands import (
    PCSOperands,
    PCSPipelineShapes,
)

from ._equivalence import assert_backend_equivalence


def test_pcs_operands_are_views_over_precomputed_model_data(
    make_pcs_model: Callable[[str], Any],
) -> None:
    """Expose only the constant-strain arrays required by Warp execution."""

    model = make_pcs_model("jax")
    operands = PCSOperands.from_model(model)

    assert operands.is_planar is False
    assert operands.num_segments == model.num_segments
    assert operands.num_dofs == model.num_dofs
    assert operands.num_gauss_points == 5
    assert operands.block_dim == model.backend_params.warp_block_dim
    assert operands.active_strain_indices is model.active_strain_indices
    assert operands.reference_strain is model.xi_ref
    assert operands.weighted_mass_diagonals is model.weighted_mass_diagonals


def test_spatial_pcs_pipeline_shapes_cover_workspace_and_results(
    make_pcs_model: Callable[[str], Any],
) -> None:
    """Describe every external allocation needed by the spatial PCS pipeline."""

    model = make_pcs_model("jax")
    operands = PCSOperands.from_model(model)
    shapes = PCSPipelineShapes.from_operands(operands, batch_size=3)

    assert shapes.spatial_dim == 6
    assert shapes.operator_outputs()["adjoint_inverse"] == (
        3 * model.num_segments * 6 * 6,
        6,
    )
    assert shapes.chain_outputs()["velocity_first"] == (3 * 6, 1)
    assert shapes.chain_outputs()["inertia"] == (
        3,
        model.num_dofs,
        model.num_dofs,
    )


@pytest.mark.parametrize(
    ("batch_size", "error_type"),
    [(True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)],
)
def test_pcs_pipeline_shapes_require_positive_integer_batch(
    make_pcs_model: Callable[[str], Any],
    batch_size: Any,
    error_type: type[Exception],
) -> None:
    """Reject allocation contracts that cannot represent an environment batch."""

    model = make_pcs_model("jax")
    operands = PCSOperands.from_model(model)
    with pytest.raises(error_type, match="batch_size"):
        PCSPipelineShapes.from_operands(operands, batch_size=batch_size)


def test_pcs_public_dynamics_apis_match_jax_on_gpu(
    make_pcs_model: Callable[[str], Any],
    state_batch: Callable[[Any], tuple[Any, Any, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Match PCS terms and forward dynamics for every public input form."""

    if jax.default_backend() != "gpu":
        pytest.skip("PCS Warp equivalence requires a JAX GPU backend.")
    pytest.importorskip("warp")
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "pcs-warp-cache"))
    jax_model = make_pcs_model("jax")
    warp_model = make_pcs_model("warp")
    q, qd, y = state_batch(jax_model)

    assert_backend_equivalence(jax_model, warp_model, q, qd, y)
