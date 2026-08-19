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
    """No averaging across starts: each descends on its own trajectory."""
    history = _run(_quadratic, [1.0, 2.0, 3.0])
    losses = history.loss_history()
    assert losses.shape == (6, BATCH)
    assert bool(jnp.all(jnp.diff(losses, axis=0) < 0))
    assert history.best_batch() == 0  # the smallest start stays best throughout


def test_history_pairs_each_loss_with_the_gains_it_was_measured_at():
    """Issue #129, per start: history[i] must not be one update ahead.

    Entry 0 must therefore still hold the untouched initialization -- the defect
    that made the legacy archives unable to report their own starting gains.
    """
    scales = [1.0, 2.0, 3.0]
    history = _run(_quadratic, scales)
    assert jnp.allclose(
        history.opt_vars[0]["opt_ctr_params"]["Kp"],
        _initial_vars(scales)["opt_ctr_params"]["Kp"],
    )
    for iteration, (loss, opt_vars) in enumerate(zip(history.loss, history.opt_vars)):
        recomputed = jax.vmap(lambda v: _quadratic(v)[0])(opt_vars)
        assert jnp.allclose(recomputed, loss), f"mismatch at iteration {iteration}"


def test_a_diverging_start_freezes_but_keeps_the_iterates_it_earned():
    """One failure must neither abort the run nor erase that start's good work."""
    # Start 0 begins smallest and crosses the poison threshold first.
    history = _run(_poisoned(0.9), [1.0, 5.0, 6.0], num_iters=6)
    mask = history.mask_history()
    assert mask.shape == (6, BATCH)
    assert not bool(jnp.all(mask[:, 0])), "start 0 was expected to die"
    assert bool(jnp.all(mask[:, 1:])), "surviving starts must stay valid"
    assert len(history) == 6, "the survivors keep iterating"
    # Start 0 was genuinely best before it diverged, so it still wins selection.
    assert history.best_batch() == 0
    assert int(history.best_iteration[0]) == 0
    for start in range(BATCH):
        best = int(history.best_iteration[start])
        assert bool(mask[best, start]), "best iteration must be a valid entry"


def test_a_start_invalid_from_its_first_evaluation_is_excluded():
    history = _run(_poisoned(2.0), [1.0, 5.0, 6.0], num_iters=4)
    assert history.dead_starts() == [0]
    assert not bool(history.mask_history()[:, 0].any())
    assert history.best_batch() != 0
    assert bool(jnp.isinf(history.best_loss[0]))


def test_only_two_rollout_snapshots_are_kept_per_start():
    """Storing one rollout per iteration would reach gigabytes at B=6, 100 iters."""
    history = _run(_quadratic, [1.0, 2.0, 3.0])
    assert history.init_aux["q_ts"].shape == (BATCH, 4, 2)
    assert history.best_aux["q_ts"].shape == (BATCH, 4, 2)
    assert not hasattr(history, "aux"), "no per-iteration rollout accumulator"
    # init_aux is iteration zero; best_aux is each start's own lowest-loss rollout.
    assert jnp.allclose(history.init_aux["q_ts"][:, 0, 0], history.loss[0])
    assert bool(jnp.all(history.best_iteration == len(history) - 1))


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
