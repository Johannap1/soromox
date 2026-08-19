"""Plot and Viser-renderer regressions for Section Vd."""

import sys
from pathlib import Path

import jax
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

SECTION_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secVd_control_gain_optimization"
)
CODE_DIR = SECTION_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import plot_control_gain_optimization as plotter  # noqa: E402
import render_control_gain_optimization_animations as renderer  # noqa: E402
from secvd_case import build_sec_vd_robot  # noqa: E402
from secvd_results import load_results  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _archive(batch, best_batch, *, method, timesteps=4):
    """Build a minimal schema-v2 archive for the plotter."""
    t = np.linspace(0, 1, timesteps)
    loss = np.ones((5, batch)) * np.arange(1, batch + 1)[None, :]
    # Make the requested start the lowest-loss one.
    loss[:, best_batch] = 0.5
    data = {
        "history_loss": loss,
        "history_finite_mask": np.ones((5, batch), dtype=bool),
        "t_ts": t,
        "best_batch": np.asarray(best_batch),
    }
    if method == "collocated":
        traj = np.zeros((batch, timesteps, 6))
        traj[:, :, :3] = np.arange(batch)[:, None, None]
        data |= {
            "q_ts_init": traj,
            "q_ts_best": traj,
            "q_des_ts": np.zeros((timesteps, 6)),
        }
    else:
        pose = np.zeros((batch, timesteps, 6))
        pose[..., :3] = 99.0
        pose[..., 3:6] = np.arange(batch)[:, None, None]
        data |= {
            "x_ts_init": pose,
            "x_ts_best": pose,
            "x_des_ts": np.zeros((timesteps, 6)),
        }
    return data


def test_plot_loss_draws_a_band_only_when_there_are_several_starts():
    figure, axis = matplotlib.pyplot.subplots()
    plotter.plot_loss(axis, np.arange(5.0)[:, None], np.ones((5, 1), bool), "A")
    assert len(axis.collections) == 0, "a single start has no spread to shade"
    matplotlib.pyplot.close(figure)

    figure, axis = matplotlib.pyplot.subplots()
    plotter.plot_loss(
        axis, np.tile(np.arange(5.0)[:, None], 6), np.ones((5, 6), bool), "A"
    )
    assert len(axis.collections) == 1
    matplotlib.pyplot.close(figure)


def test_frozen_starts_do_not_drag_the_loss_band():
    """A masked-out entry is excluded from the band, not plotted as NaN."""
    loss = np.tile(np.arange(1.0, 6.0)[:, None], 3)
    loss[3:, 2] = np.nan
    mask = np.ones((5, 3), dtype=bool)
    mask[3:, 2] = False
    figure, axis = matplotlib.pyplot.subplots()
    plotter.plot_loss(axis, loss, mask, "A")
    assert np.all(np.isfinite(axis.lines[0].get_ydata()))
    matplotlib.pyplot.close(figure)


def test_each_panel_uses_its_own_best_batch(monkeypatch):
    """Issue #128: the synergistic panel must not inherit the collocated index."""
    collocated = _archive(6, best_batch=1, method="collocated")
    synergistic = _archive(6, best_batch=4, method="synergistic")
    monkeypatch.setattr(
        plotter,
        "load_results",
        lambda path, expected_method: (
            collocated if expected_method == "collocated" else synergistic
        ),
    )
    figure = plotter.build_figure(Path("unused"))
    # Each channel draws target, initial median, then the best start's solid line.
    collocated_solid = figure.axes[1].lines[2].get_ydata()
    synergistic_solid = figure.axes[3].lines[2].get_ydata()
    np.testing.assert_allclose(collocated_solid, 1.0)
    np.testing.assert_allclose(synergistic_solid, 4.0)
    matplotlib.pyplot.close(figure)


def test_plotter_uses_synergistic_position_channels(monkeypatch):
    collocated = _archive(1, best_batch=0, method="collocated")
    synergistic = _archive(1, best_batch=0, method="synergistic")
    # Orientation is 99.0 and position is 0.0; the panel must show position.
    monkeypatch.setattr(
        plotter,
        "load_results",
        lambda path, expected_method: (
            collocated if expected_method == "collocated" else synergistic
        ),
    )
    figure = plotter.build_figure(Path("unused"))
    plotted = figure.axes[3].lines[2].get_ydata()
    np.testing.assert_allclose(plotted, 0.0)
    matplotlib.pyplot.close(figure)


def test_resampling_matches_requested_fps():
    t = np.linspace(0, 1, 101)
    q = np.column_stack([t, 2 * t])
    pose = np.column_stack([t] * 6)
    frame_t, frame_q, frame_pose = renderer.resample_trajectory(t, q, pose, fps=10)
    assert len(frame_t) == 11
    np.testing.assert_allclose(frame_q[:, 0], frame_t)
    np.testing.assert_allclose(frame_pose[:, 5], frame_t)


def test_desired_shape_wireframe_is_finite_and_three_dimensional():
    z = np.linspace(0.0, 0.1, 12)
    curve = np.column_stack([0.01 * np.sin(8 * z), np.zeros_like(z), z])
    segments = renderer.make_tube_wire_segments(curve, radius=0.02)
    assert segments.ndim == 3
    assert segments.shape[1:] == (2, 3)
    assert np.all(np.isfinite(segments))


def test_method_selection_and_overwrite(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        renderer,
        "render_method",
        lambda **kwargs: (
            calls.append(kwargs["name"]) or [tmp_path / f"{kwargs['name']}.mp4"]
        ),
    )
    outputs = renderer.render_saved_results(
        data_dir=tmp_path,
        output_dir=tmp_path,
        method="synergistic",
        renderer_factory=object,
    )
    assert calls == ["synergistic"]
    assert outputs == [tmp_path / "synergistic.mp4"]

    (tmp_path / "collocated_tracking.mp4").touch()
    with pytest.raises(FileExistsError, match="--force"):
        renderer.render_saved_results(
            data_dir=tmp_path,
            output_dir=tmp_path,
            method="both",
            renderer_factory=object,
        )
    assert calls == ["synergistic"], "preflight must fail before either render starts"


def test_mock_viser_receives_robot_trail_target_and_paths(monkeypatch, tmp_path):
    timesteps = 5
    data = {
        "t_ts": np.linspace(0, 1, timesteps),
        "best_batch": np.asarray(0),
        "q_ts_best": np.zeros((1, timesteps, 6)),
        "x_ts_best": np.zeros((1, timesteps, 6)),
        "x_des_ts": np.zeros((timesteps, 6)),
    }
    monkeypatch.setattr(renderer, "load_results", lambda *_args, **_kwargs: data)
    monkeypatch.setattr(renderer, "build_sec_vd_robot", lambda: (object(), None, 0.1))
    monkeypatch.setattr(renderer, "validate_pose_consistency", lambda *_args: None)
    captured = {}

    class FakeRenderer:
        def __init__(self, robot, **kwargs):
            captured["init"] = (robot, kwargs)

        def render_sequence(self, **kwargs):
            captured["render"] = kwargs

    outputs = renderer.render_method(
        name="synergistic",
        data_dir=tmp_path,
        output_dir=tmp_path,
        fps=4,
        make_gif=False,
        force=False,
        port=8080,
        open_browser=False,
        record_client_timeout=2.0,
        record_frame_timeout=3.0,
        renderer_factory=FakeRenderer,
    )
    assert outputs == [tmp_path / "synergistic_tracking.mp4"]
    assert captured["render"]["q_ts"].shape == (5, 6)
    assert captured["init"][1]["desired_q_ts"] is None
    backbone_colors = captured["init"][1]["color_config"].backbone.point_colors
    np.testing.assert_allclose(
        backbone_colors[:, 3], renderer.SYNERGISTIC_ROBOT_OPACITY
    )
    assert captured["render"]["dynamic_spheres_positions"].shape == (2, 5, 3)
    np.testing.assert_allclose(
        captured["render"]["dynamic_spheres_radii"],
        [renderer.MARKER_RADIUS, renderer.TARGET_MARKER_RADIUS],
    )
    np.testing.assert_allclose(
        captured["render"]["dynamic_spheres_colors"],
        np.stack([renderer.CURRENT_POSITION_COLOR, renderer.TARGET_COLOR]),
    )
    assert captured["render"]["static_spheres_positions"] is None
    assert captured["render"]["record_path"].endswith("synergistic_tracking.mp4")

    outputs[0].touch()
    with pytest.raises(FileExistsError, match="--force"):
        renderer.render_method(
            name="synergistic",
            data_dir=tmp_path,
            output_dir=tmp_path,
            fps=4,
            make_gif=False,
            force=False,
            port=8080,
            open_browser=False,
            record_client_timeout=2.0,
            record_frame_timeout=3.0,
            renderer_factory=FakeRenderer,
        )


def test_synergistic_trail_contains_only_time_varying_reference(monkeypatch, tmp_path):
    timesteps = 5
    reference_pose = np.zeros((timesteps, 6))
    reference_pose[:, 3] = np.linspace(0.0, 0.04, timesteps)
    data = {
        "t_ts": np.linspace(0, 1, timesteps),
        "best_batch": np.asarray(0),
        "q_ts_best": np.zeros((1, timesteps, 6)),
        "x_ts_best": np.zeros((1, timesteps, 6)),
        "x_des_ts": reference_pose,
    }
    monkeypatch.setattr(renderer, "load_results", lambda *_args, **_kwargs: data)
    monkeypatch.setattr(renderer, "build_sec_vd_robot", lambda: (object(), None, 0.1))
    monkeypatch.setattr(renderer, "validate_pose_consistency", lambda *_args: None)
    captured = {}

    class FakeRenderer:
        def __init__(self, _robot, **_kwargs):
            pass

        def render_sequence(self, **kwargs):
            captured.update(kwargs)

    renderer.render_method(
        name="synergistic",
        data_dir=tmp_path,
        output_dir=tmp_path,
        fps=4,
        make_gif=False,
        force=False,
        port=8080,
        open_browser=False,
        record_client_timeout=2.0,
        record_frame_timeout=3.0,
        renderer_factory=FakeRenderer,
    )
    expected_reference = reference_pose[:, 3:6]
    np.testing.assert_allclose(captured["static_spheres_positions"], expected_reference)
    np.testing.assert_allclose(
        captured["static_spheres_colors"],
        np.tile(renderer.TARGET_TRAIL_COLOR, (timesteps, 1)),
    )


def test_renderer_animates_the_archives_own_best_start(monkeypatch, tmp_path):
    """Issue #128, renderer side: never assume the best start is index 0."""
    timesteps, batch, best = 5, 4, 2
    q_ts_best = np.zeros((batch, timesteps, 6))
    q_ts_best[best, :, :] = 7.0
    data = {
        "t_ts": np.linspace(0, 1, timesteps),
        "best_batch": np.asarray(best),
        "q_ts_best": q_ts_best,
        "x_ts_best": np.zeros((batch, timesteps, 6)),
        "x_des_ts": np.zeros((timesteps, 6)),
    }
    monkeypatch.setattr(renderer, "load_results", lambda *_args, **_kwargs: data)
    monkeypatch.setattr(renderer, "build_sec_vd_robot", lambda: (object(), None, 0.1))
    monkeypatch.setattr(renderer, "validate_pose_consistency", lambda *_args: None)
    captured = {}

    class FakeRenderer:
        def __init__(self, _robot, **kwargs):
            captured["init"] = kwargs

        def render_sequence(self, **kwargs):
            captured["render"] = kwargs

    renderer.render_method(
        name="synergistic",
        data_dir=tmp_path,
        output_dir=tmp_path,
        fps=4,
        make_gif=False,
        force=False,
        port=8080,
        open_browser=False,
        record_client_timeout=2.0,
        record_frame_timeout=3.0,
        renderer_factory=FakeRenderer,
    )
    np.testing.assert_allclose(captured["render"]["q_ts"], 7.0)


def test_collocated_render_receives_configuration_target(monkeypatch, tmp_path):
    timesteps = 5
    data = {
        "t_ts": np.linspace(0, 1, timesteps),
        "best_batch": np.asarray(0),
        "q_ts_best": np.zeros((1, timesteps, 6)),
        "q_des_ts": np.ones((timesteps, 6)),
        "x_ts_best": np.zeros((1, timesteps, 6)),
        "x_des_ts": np.zeros((timesteps, 6)),
    }
    monkeypatch.setattr(renderer, "load_results", lambda *_args, **_kwargs: data)
    monkeypatch.setattr(renderer, "build_sec_vd_robot", lambda: (object(), None, 0.1))
    monkeypatch.setattr(renderer, "validate_pose_consistency", lambda *_args: None)
    captured = {}

    class FakeRenderer:
        def __init__(self, _robot, **kwargs):
            captured["init"] = kwargs

        def render_sequence(self, **kwargs):
            captured["render"] = kwargs

    renderer.render_method(
        name="collocated",
        data_dir=tmp_path,
        output_dir=tmp_path,
        fps=4,
        make_gif=False,
        force=False,
        port=8080,
        open_browser=False,
        record_client_timeout=2.0,
        record_frame_timeout=3.0,
        renderer_factory=FakeRenderer,
    )
    assert captured["init"]["desired_q_ts"].shape == (5, 6)
    np.testing.assert_allclose(
        captured["init"]["color_config"].backbone.point_colors[:, 3], 1.0
    )
    assert captured["render"]["static_spheres_positions"] is None
    assert captured["render"]["dynamic_spheres_positions"] is None


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The committed placeholders are schema-v1, single-start archives that the "
        "v2 validator rejects. They are regenerated as six-start runs at the end "
        "of the batching work; remove this marker with that regeneration."
    ),
)
@pytest.mark.parametrize("method", ["collocated", "synergistic"])
def test_committed_pose_matches_forward_kinematics(method):
    data = load_results(SECTION_DIR / "data" / method, expected_method=method)
    assert data["history_loss"].shape[1] == int(data["batch_size"])
    assert int(data["completed_iterations"]) == 5
    assert bool(data["is_placeholder"])
    robot, _routing, total_length = build_sec_vd_robot()
    renderer.validate_pose_consistency(robot, total_length, data)
