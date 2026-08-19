"""Tests for the one supported Section Vd archive schema."""

import sys
from pathlib import Path

import numpy as np
import pytest

SECTION_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secVd_control_gain_optimization"
)
CODE_DIR = SECTION_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from secvd_results import (  # noqa: E402
    best_batch_from_loss,
    best_iteration_per_batch,
    improvement_over_initial_median,
    load_results,
    save_optimization_outputs,
    validate_results,
)

ITERATIONS, BATCH, TIMESTEPS, DOFS, ACTUATORS = 5, 6, 7, 6, 3


def _descending_loss() -> np.ndarray:
    """Losses that fall monotonically, with start 3 ending lowest."""
    start_offsets = np.array([5.0, 7.0, 6.0, 4.0, 6.5, 8.0])
    decay = np.arange(ITERATIONS, dtype=float)[:, None] * 0.1
    return start_offsets[None, :] - decay


def _write_results(
    tmp_path: Path,
    method: str = "synergistic",
    history_loss: np.ndarray | None = None,
    finite_mask: np.ndarray | None = None,
) -> Path:
    loss = _descending_loss() if history_loss is None else history_loss
    mask = np.ones_like(loss, dtype=bool) if finite_mask is None else finite_mask
    q = np.zeros((BATCH, TIMESTEPS, DOFS))
    pose = np.zeros((BATCH, TIMESTEPS, 6))
    controls = np.zeros((BATCH, TIMESTEPS, ACTUATORS))
    gains = {name: np.ones((BATCH, ACTUATORS)) for name in ("Kp", "Ki", "Kd")}
    kwargs = {}
    if method == "collocated":
        kwargs["q_des_ts"] = np.zeros((TIMESTEPS, DOFS))
    return save_optimization_outputs(
        tmp_path,
        method=method,
        batch_size=BATCH,
        history_loss=loss,
        history_time=np.ones(ITERATIONS),
        history_finite_mask=mask,
        opt_vars_history=[
            {
                "opt_ctr_params": {
                    name: np.ones((BATCH, ACTUATORS)) * i for name in ("Kp", "Ki", "Kd")
                }
            }
            for i in range(ITERATIONS)
        ],
        init_gains=gains,
        init_seed=0,
        init_scheme="log_uniform_v1",
        init_spread=3.0,
        t_ts=np.linspace(0, 1, TIMESTEPS),
        q_ts_init=q,
        q_ts_best=q,
        qd_ts_init=q,
        qd_ts_best=q,
        u_ts_init=controls,
        u_ts_best=controls,
        x_ts_init=pose,
        x_ts_best=pose,
        x_des_ts=np.zeros((TIMESTEPS, 6)),
        best_iteration=best_iteration_per_batch(loss, mask),
        best_batch=best_batch_from_loss(loss, mask),
        is_placeholder=True,
        **kwargs,
    )


def test_batch_axis_appears_only_where_data_varies_per_start(tmp_path):
    data = load_results(_write_results(tmp_path), expected_method="synergistic")
    assert data["history_loss"].shape == (ITERATIONS, BATCH)
    assert data["history_Kp"].shape == (ITERATIONS, BATCH, ACTUATORS)
    assert data["init_Kp"].shape == (BATCH, ACTUATORS)
    assert data["q_ts_best"].shape == (BATCH, TIMESTEPS, DOFS)
    assert data["x_ts_best"].shape == (BATCH, TIMESTEPS, 6)
    assert data["best_iteration"].shape == (BATCH,)
    # Shared by every start, so no batch axis at all.
    assert data["t_ts"].shape == (TIMESTEPS,)
    assert data["x_des_ts"].shape == (TIMESTEPS, 6)
    assert data["history_time"].shape == (ITERATIONS,)
    assert int(data["batch_size"]) == BATCH
    assert int(data["completed_iterations"]) == ITERATIONS
    assert bool(data["is_placeholder"])


def test_per_iteration_trajectory_histories_are_not_archived(tmp_path):
    """They would reach gigabytes at 100 iterations and six starts."""
    data = load_results(_write_results(tmp_path))
    assert "history_qd_ts" not in data
    assert "history_u_ts" not in data


def test_initialization_metadata_round_trips(tmp_path):
    data = load_results(_write_results(tmp_path))
    assert int(data["init_seed"]) == 0
    assert str(data["init_scheme"].item()) == "log_uniform_v1"
    assert float(data["init_spread"]) == 3.0


def test_selection_indices_are_recorded_and_consistent(tmp_path):
    data = load_results(_write_results(tmp_path))
    assert int(data["best_batch"]) == 3
    assert np.array_equal(data["best_iteration"], np.full(BATCH, ITERATIONS - 1))


def test_schema_v1_archive_is_rejected_clearly(tmp_path):
    path = tmp_path / "optimization_results.npz"
    np.savez(path, history_loss=np.zeros((5, 1)))
    with pytest.raises(ValueError, match="not supported"):
        load_results(path)


def test_batch_size_must_match_the_loss_axis(tmp_path):
    data = load_results(_write_results(tmp_path))
    data["batch_size"] = np.asarray(BATCH + 1, dtype=np.int64)
    with pytest.raises(ValueError, match="batch_size must equal"):
        validate_results(data)


def test_selection_indices_are_checked_against_the_loss_history(tmp_path):
    data = load_results(_write_results(tmp_path))
    data["best_batch"] = np.asarray(0, dtype=np.int64)
    with pytest.raises(ValueError, match="best_batch disagrees"):
        validate_results(data)

    data = load_results(_write_results(tmp_path))
    data["best_iteration"][2] = 0
    with pytest.raises(ValueError, match="best_iteration disagrees"):
        validate_results(data)


def test_out_of_range_best_iteration_is_rejected(tmp_path):
    data = load_results(_write_results(tmp_path))
    data["best_iteration"][0] = ITERATIONS
    with pytest.raises(ValueError, match="must lie in"):
        validate_results(data)


def test_masked_entries_may_be_non_finite_but_results_may_not(tmp_path):
    """A frozen start keeps the archive valid; a bad trajectory does not."""
    loss = _descending_loss()
    mask = np.ones_like(loss, dtype=bool)
    mask[3:, 4] = False
    loss[3:, 4] = np.nan
    data = load_results(_write_results(tmp_path, finite_mask=mask, history_loss=loss))
    validate_results(data)
    assert not bool(data["history_finite_mask"][-1, 4])

    data["q_ts_best"][0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="q_ts_best contains non-finite"):
        validate_results(data)


def test_unmasked_non_finite_loss_is_rejected(tmp_path):
    data = load_results(_write_results(tmp_path))
    data["history_loss"][0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite where history_finite_mask"):
        validate_results(data)


def test_start_that_never_produced_a_finite_iterate_is_rejected(tmp_path):
    loss = _descending_loss()
    mask = np.ones_like(loss, dtype=bool)
    mask[:, 2] = False
    with pytest.raises(ValueError, match="never produced a finite iterate"):
        _write_results(tmp_path, finite_mask=mask, history_loss=loss)


def test_collocated_requires_its_configuration_reference(tmp_path):
    data = load_results(
        _write_results(tmp_path, method="collocated"), expected_method="collocated"
    )
    assert data["q_des_ts"].shape == (TIMESTEPS, DOFS)
    del data["q_des_ts"]
    with pytest.raises(ValueError, match="require q_des_ts"):
        validate_results(data)


def test_best_batch_is_derived_per_archive(tmp_path):
    """Issue #128: each method must select from its own losses."""
    collocated_loss = _descending_loss()
    synergistic_loss = _descending_loss()[:, ::-1].copy()
    assert best_batch_from_loss(collocated_loss) == 3
    assert best_batch_from_loss(synergistic_loss) == BATCH - 1 - 3


def test_improvement_metric_matches_the_recovered_paper_formula():
    loss = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5]])
    # median of [1, 2, 3, 4] is 2.5; best final loss is 0.5.
    assert improvement_over_initial_median(loss) == pytest.approx(1.0 - 0.5 / 2.5)
