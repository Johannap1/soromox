"""Tests for the reproducible Section Vd multi-start gain initialization."""

import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

SECTION_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secVd_control_gain_optimization"
)
CODE_DIR = SECTION_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from secvd_init import (  # noqa: E402
    GAIN_ORDER,
    LEGACY_UNIFORM_BOX,
    SECVD_BATCH_SIZE,
    describe_initial_gains,
    sample_initial_gains,
)

NOMINAL = {"Kp": 5e1 * jnp.ones(3), "Ki": 5e0 * jnp.ones(3), "Kd": 1e0 * jnp.ones(3)}


def test_first_start_is_exactly_the_nominal_set():
    """A single-start run must stay a strict subset of a batched one."""
    for scheme in ("log_uniform_v1", "legacy_uniform"):
        gains = sample_initial_gains(NOMINAL, scheme=scheme)
        for name in GAIN_ORDER:
            assert jnp.array_equal(gains[name][0], NOMINAL[name])


def test_default_batch_matches_the_legacy_archive_width():
    gains = sample_initial_gains(NOMINAL)
    for name in GAIN_ORDER:
        assert gains[name].shape == (SECVD_BATCH_SIZE, 3)


def test_batch_size_one_returns_only_the_nominal():
    gains = sample_initial_gains(NOMINAL, batch_size=1)
    for name in GAIN_ORDER:
        assert gains[name].shape == (1, 3)
        assert jnp.array_equal(gains[name][0], NOMINAL[name])


def test_sampling_is_reproducible_and_seed_dependent():
    first = sample_initial_gains(NOMINAL, seed=7)
    again = sample_initial_gains(NOMINAL, seed=7)
    other = sample_initial_gains(NOMINAL, seed=8)
    for name in GAIN_ORDER:
        assert jnp.array_equal(first[name], again[name])
        # Entry 0 is the nominal under every seed; the perturbed ones must move.
        assert not jnp.array_equal(first[name][1:], other[name][1:])


def test_log_uniform_samples_stay_inside_the_documented_box():
    spread = 3.0
    gains = sample_initial_gains(NOMINAL, spread=spread)
    for name in GAIN_ORDER:
        nominal = NOMINAL[name]
        assert bool(jnp.all(gains[name] >= nominal / spread - 1e-9))
        assert bool(jnp.all(gains[name] <= nominal * spread + 1e-9))


def test_legacy_uniform_samples_stay_inside_the_recovered_box():
    gains = sample_initial_gains(NOMINAL, scheme="legacy_uniform", seed=35)
    for name in GAIN_ORDER:
        low, high = LEGACY_UNIFORM_BOX[name]
        assert bool(jnp.all(gains[name][1:] >= low))
        assert bool(jnp.all(gains[name][1:] <= high))


def test_each_start_is_one_scalar_per_gain():
    """Issue #154 describes each start as three numbers, not 3 x m numbers."""
    gains = sample_initial_gains(NOMINAL)
    for name in GAIN_ORDER:
        assert bool(jnp.all(gains[name] == gains[name][:, :1]))


def test_log_uniform_preserves_anisotropy_of_the_nominal():
    nominal = {
        "Kp": jnp.array([10.0, 20.0, 40.0]),
        "Ki": 5e0 * jnp.ones(3),
        "Kd": 1e0 * jnp.ones(3),
    }
    gains = sample_initial_gains(nominal)
    ratios = gains["Kp"] / nominal["Kp"]
    assert bool(jnp.allclose(ratios, ratios[:, :1]))


def test_gain_keys_are_consumed_in_a_fixed_order():
    """Reordering the input dict must not change which key each gain draws."""
    reordered = {name: NOMINAL[name] for name in reversed(GAIN_ORDER)}
    baseline = sample_initial_gains(NOMINAL)
    shuffled = sample_initial_gains(reordered)
    for name in GAIN_ORDER:
        assert jnp.array_equal(baseline[name], shuffled[name])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"scheme": "not_a_scheme"},
        {"spread": 1.0},
        {"spread": 0.5},
    ],
)
def test_invalid_requests_are_rejected(kwargs):
    with pytest.raises(ValueError):
        sample_initial_gains(NOMINAL, **kwargs)


def test_unknown_gain_keys_are_rejected():
    with pytest.raises(ValueError):
        sample_initial_gains({"Kp": jnp.ones(3), "Ki": jnp.ones(3)})


def test_non_positive_nominal_is_rejected_for_log_uniform():
    nominal = dict(NOMINAL, Ki=jnp.zeros(3))
    with pytest.raises(ValueError):
        sample_initial_gains(nominal)
    # The absolute-box scheme has no such restriction.
    sample_initial_gains(nominal, scheme="legacy_uniform")


def test_description_lists_every_start():
    table = describe_initial_gains(sample_initial_gains(NOMINAL))
    assert all(name in table for name in GAIN_ORDER)
    assert len(table.splitlines()) == SECVD_BATCH_SIZE + 3
