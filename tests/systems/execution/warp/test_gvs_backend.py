"""JAX/Warp equivalence tests for shape-generic GVS dynamics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import pytest
from numpy.testing import assert_allclose

from soromox.systems.execution.warp.gvs.operands import (
    GVSOperands,
    GVSPipelineShapes,
)

from ._equivalence import assert_backend_equivalence


def test_gvs_operands_are_views_over_precomputed_model_data(
    make_gvs_model: Callable[[str], Any],
) -> None:
    """Keep preprocessing in the model and the executor contract minimal."""

    model = make_gvs_model("jax")
    operands = GVSOperands.from_model(model, block_dim=128)

    assert operands.num_segments == model.num_segments
    assert operands.num_dofs == model.num_dofs
    assert operands.num_cells == model.max_num_integration_points - 1
    assert operands.num_quadrature == model.max_num_integration_points - 2
    assert operands.block_dim == 128
    assert operands.joint_basis is model.B_joint
    assert operands.link_basis_z1_values is model.scaled_B_Z1_values
    assert operands.weighted_mass_diagonals is model.inner_weighted_mass_diagonals
    assert_allclose(operands.gravity_base, model.gravity_base, rtol=0.0, atol=0.0)


def test_gvs_pipeline_shapes_cover_public_workspace_and_outputs(
    make_gvs_model: Callable[[str], Any],
) -> None:
    """Provide external Warp solvers one allocation contract for fixed topology."""

    model = make_gvs_model("jax")
    operands = GVSOperands.from_model(model, block_dim=128)
    shapes = GVSPipelineShapes.from_operands(operands, batch_size=3)

    assert shapes.local_state == (3 * model.num_segments, model.max_dof)
    assert shapes.joint_outputs()["adjoint"] == (3 * model.num_segments * 6, 6)
    assert shapes.cell_outputs()["tangent_local"] == (
        3 * model.num_segments * operands.num_cells * 6,
        model.max_dof,
    )
    assert shapes.chain_outputs()["inertia"] == (
        3,
        model.num_dofs,
        model.num_dofs,
    )
    assert shapes.chain_outputs()["gravity_force"] == (3, model.num_dofs)


@pytest.mark.parametrize(
    ("batch_size", "error_type"),
    [(True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)],
)
def test_gvs_pipeline_shapes_require_positive_integer_batch(
    make_gvs_model: Callable[[str], Any],
    batch_size: Any,
    error_type: type[Exception],
) -> None:
    """Reject allocation contracts that cannot represent an environment batch."""

    model = make_gvs_model("jax")
    operands = GVSOperands.from_model(model, block_dim=128)
    with pytest.raises(error_type, match="batch_size"):
        GVSPipelineShapes.from_operands(operands, batch_size=batch_size)


def test_gvs_public_dynamics_apis_match_jax_on_gpu(
    make_gvs_model: Callable[[str], Any],
    state_batch: Callable[[Any], tuple[Any, Any, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Match GVS terms and forward dynamics for every public input form."""

    if jax.default_backend() != "gpu":
        pytest.skip("GVS Warp equivalence requires a JAX GPU backend.")
    pytest.importorskip("warp")
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "gvs-warp-cache"))
    jax_model = make_gvs_model("jax")
    warp_model = make_gvs_model("warp")
    q, qd, y = state_batch(jax_model)

    assert_backend_equivalence(jax_model, warp_model, q, qd, y)
