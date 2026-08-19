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


@pytest.mark.parametrize("scheme", ["log_uniform_v1", "legacy_uniform"])
@pytest.mark.parametrize("batch_size", [1, SECVD_BATCH_SIZE])
def test_start_zero_is_always_the_untouched_nominal(scheme, batch_size):
    """A single-start run must stay a strict subset of a batched one."""
    gains = sample_initial_gains(NOMINAL, batch_size=batch_size, scheme=scheme)
    for name in GAIN_ORDER:
        assert gains[name].shape == (batch_size, 3)
        assert jnp.array_equal(gains[name][0], NOMINAL[name])


def test_default_batch_matches_the_legacy_archive_width():
    assert SECVD_BATCH_SIZE == 6
    assert sample_initial_gains(NOMINAL)["Kp"].shape == (6, 3)


def test_sampling_is_reproducible_and_seed_dependent():
    first = sample_initial_gains(NOMINAL, seed=7)
    again = sample_initial_gains(NOMINAL, seed=7)
    other = sample_initial_gains(NOMINAL, seed=8)
    for name in GAIN_ORDER:
        assert jnp.array_equal(first[name], again[name])
        # Start 0 is nominal under every seed; only the sampled ones may differ.
        assert not jnp.array_equal(first[name][1:], other[name][1:])


def test_samples_stay_inside_their_scheme_box():
    spread = 3.0
    log_uniform = sample_initial_gains(NOMINAL, spread=spread)
    legacy = sample_initial_gains(NOMINAL, scheme="legacy_uniform", seed=35)
    for name in GAIN_ORDER:
        assert bool(jnp.all(log_uniform[name] >= NOMINAL[name] / spread - 1e-9))
        assert bool(jnp.all(log_uniform[name] <= NOMINAL[name] * spread + 1e-9))
        low, high = LEGACY_UNIFORM_BOX[name]
        assert bool(jnp.all((legacy[name][1:] >= low) & (legacy[name][1:] <= high)))


def test_each_start_is_one_scalar_factor_per_gain():
    """Issue #154 describes each start as three numbers, not 3 x m numbers.

    Scaling the whole nominal vector also preserves any anisotropy it carries,
    which per-component sampling would destroy.
    """
    isotropic = sample_initial_gains(NOMINAL)
    for name in GAIN_ORDER:
        assert bool(jnp.all(isotropic[name] == isotropic[name][:, :1]))

    anisotropic = dict(NOMINAL, Kp=jnp.array([10.0, 20.0, 40.0]))
    ratios = sample_initial_gains(anisotropic)["Kp"] / anisotropic["Kp"]
    assert bool(jnp.allclose(ratios, ratios[:, :1]))


def test_gain_keys_are_consumed_in_a_fixed_order():
    """Reordering the input dict must not change which key each gain draws."""
    reordered = {name: NOMINAL[name] for name in reversed(GAIN_ORDER)}
    baseline = sample_initial_gains(NOMINAL)
    shuffled = sample_initial_gains(reordered)
    for name in GAIN_ORDER:
        assert jnp.array_equal(baseline[name], shuffled[name])


@pytest.mark.parametrize(
    ("nominal", "kwargs"),
    [
        (NOMINAL, {"batch_size": 0}),
        (NOMINAL, {"scheme": "not_a_scheme"}),
        (NOMINAL, {"spread": 1.0}),
        (NOMINAL, {"spread": 0.5}),
        ({"Kp": jnp.ones(3), "Ki": jnp.ones(3)}, {}),
        # log_uniform_v1 is multiplicative, so a zero gain has no neighbourhood.
        (dict(NOMINAL, Ki=jnp.zeros(3)), {}),
    ],
)
def test_requests_that_cannot_produce_finite_gains_are_rejected(nominal, kwargs):
    with pytest.raises(ValueError):
        sample_initial_gains(nominal, **kwargs)


def test_absolute_box_scheme_accepts_a_zero_nominal():
    gains = sample_initial_gains(
        dict(NOMINAL, Ki=jnp.zeros(3)), scheme="legacy_uniform"
    )
    assert bool(jnp.all(gains["Ki"][1:] > 0.0))


def test_description_lists_every_start():
    table = describe_initial_gains(sample_initial_gains(NOMINAL))
    assert all(name in table for name in GAIN_ORDER)
    assert len(table.splitlines()) == SECVD_BATCH_SIZE + 3
