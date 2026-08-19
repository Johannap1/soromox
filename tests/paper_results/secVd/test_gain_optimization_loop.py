"""Tests for the shared multi-start Section Vd optimization loop."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pytest

SECTION_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secVd_control_gain_optimization"
)
CODE_DIR = SECTION_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from gain_optimization_loop import run_gain_optimization  # noqa: E402

jax.config.update("jax_enable_x64", True)

BATCH = 3


def _quadratic(opt_vars):
    """A toy loss with its minimum at zero, plus a rollout-shaped aux."""
    gains = opt_vars["opt_ctr_params"]
    loss = sum(jnp.sum(gains[name] ** 2) for name in ("Kp", "Ki", "Kd"))
    return loss, {"q_ts": jnp.ones((4, 2)) * loss, "t_ts": jnp.arange(4.0)}


def _poisoned(threshold):
    """Like ``_quadratic`` but non-finite once a start's Kp falls below ``threshold``."""

    def evaluate(opt_vars):
        loss, aux = _quadratic(opt_vars)
        poison = jnp.where(
            opt_vars["opt_ctr_params"]["Kp"][0] < threshold, jnp.nan, 0.0
        )
        return loss + poison, aux

    return evaluate


def _initial_vars(scales):
    return {
        "opt_ctr_params": {
            name: jnp.asarray(scales)[:, None] * jnp.ones((len(scales), 2))
            for name in ("Kp", "Ki", "Kd")
        }
    }


def _optimizer():
    labels = {"opt_ctr_params": {"Kp": "P", "Ki": "I", "Kd": "D"}}
    return optax.multi_transform(
        {"P": optax.sgd(0.1), "I": optax.sgd(0.1), "D": optax.sgd(0.1)}, labels
    )


def _run(evaluate, scales, num_iters=6):
    return run_gain_optimization(
        gradient_fn=jax.value_and_grad(evaluate, has_aux=True),
        optimizer=_optimizer(),
        opt_vars=_initial_vars(scales),
        num_iters=num_iters,
        batch_size=len(scales),
        progress_label="Test",
    )


def test_every_start_is_optimized_independently():
    history = _run(_quadratic, [1.0, 2.0, 3.0])
    losses = history.loss_history()
    assert losses.shape == (6, BATCH)
    # Each start descends on its own; none is averaged into the others.
    assert bool(jnp.all(losses[-1] < losses[0]))
    assert bool(jnp.all(jnp.diff(losses, axis=0) < 0))
    # The smallest initialization stays the best throughout.
    assert history.best_batch() == 0


def test_loss_is_paired_with_the_gains_it_was_measured_at():
    """Issue #129, per start: history[i] must not be one update ahead."""
    history = _run(_quadratic, [1.0, 2.0, 3.0])
    for iteration, (loss, opt_vars) in enumerate(zip(history.loss, history.opt_vars)):
        recomputed = jax.vmap(lambda v: _quadratic(v)[0])(opt_vars)
        assert jnp.allclose(recomputed, loss), f"mismatch at iteration {iteration}"


def test_first_recorded_gains_are_the_untouched_initialization():
    scales = [1.0, 2.0, 3.0]
    history = _run(_quadratic, scales)
    first = history.opt_vars[0]["opt_ctr_params"]["Kp"]
    assert jnp.allclose(first, _initial_vars(scales)["opt_ctr_params"]["Kp"])


def test_a_diverging_start_freezes_without_killing_the_others():
    # Start 0 begins smallest and crosses the poison threshold first.
    history = _run(_poisoned(0.9), [1.0, 5.0, 6.0], num_iters=6)
    mask = history.mask_history()
    assert mask.shape == (6, BATCH)
    assert not bool(jnp.all(mask[:, 0])), "start 0 was expected to die"
    assert bool(jnp.all(mask[:, 1:])), "surviving starts must stay valid"
    # The loop keeps going for the survivors rather than aborting at the failure.
    assert len(history) == 6


def test_a_start_keeps_the_valid_iterates_it_earned_before_dying():
    """Dying later must not retroactively discard a start's good iterations."""
    history = _run(_poisoned(0.9), [1.0, 5.0, 6.0], num_iters=6)
    # Start 0 was genuinely best at iteration 0 before it diverged.
    assert history.best_batch() == 0
    assert int(history.best_iteration[0]) == 0
    assert history.dead_starts() == []
    mask = history.mask_history()
    for start in range(BATCH):
        best = int(history.best_iteration[start])
        assert bool(mask[best, start]), "best iteration must be a valid entry"


def test_a_start_that_is_invalid_from_the_first_evaluation_is_excluded():
    # The threshold poisons start 0 at iteration 0, so it never earns an iterate.
    history = _run(_poisoned(2.0), [1.0, 5.0, 6.0], num_iters=4)
    assert history.dead_starts() == [0]
    assert not bool(history.mask_history()[:, 0].any())
    # Selection must fall to a start that actually produced something.
    assert history.best_batch() != 0
    assert bool(jnp.isinf(history.best_loss[0]))


def test_best_rollout_is_tracked_per_start_without_storing_every_iteration():
    history = _run(_quadratic, [1.0, 2.0, 3.0])
    assert history.init_aux["q_ts"].shape == (BATCH, 4, 2)
    assert history.best_aux["q_ts"].shape == (BATCH, 4, 2)
    # The archive only ever needs these two snapshots, not one per iteration.
    assert not hasattr(history, "aux")
    # Monotone descent means the best rollout is the last one, per start.
    assert bool(jnp.all(history.best_iteration == len(history) - 1))


def test_initial_rollout_is_the_iteration_zero_one():
    history = _run(_quadratic, [1.0, 2.0, 3.0])
    initial_loss = history.loss[0]
    assert jnp.allclose(history.init_aux["q_ts"][:, 0, 0], initial_loss)


def test_single_start_still_works():
    history = _run(_quadratic, [2.0], num_iters=3)
    assert history.loss_history().shape == (3, 1)
    assert history.best_batch() == 0


@pytest.mark.parametrize("bad", [{"num_iters": 0}, {"batch_size": 0}])
def test_invalid_sizes_are_rejected(bad):
    kwargs = {
        "gradient_fn": jax.value_and_grad(_quadratic, has_aux=True),
        "optimizer": _optimizer(),
        "opt_vars": _initial_vars([1.0]),
        "num_iters": 2,
        "batch_size": 1,
    }
    kwargs.update(bad)
    with pytest.raises(ValueError):
        run_gain_optimization(**kwargs)
