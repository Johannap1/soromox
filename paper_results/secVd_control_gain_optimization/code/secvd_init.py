"""Reproducible multi-start gain initialization for the Section Vd study.

The published Section Vd results came from six independently optimized gain
initializations, but neither those gains nor their seed survive in any committed
artifact (see the provenance section of this case's README). This module defines
the replacement: a named, seeded scheme whose identity is recorded in every
archive, so a future recovery of the original generator can be added as another
scheme rather than a rewrite.

Batch entry ``0`` is always the nominal gain set, which keeps a single-start run
a strict subset of a batched one. The remaining entries perturb the nominal by a
single scalar factor per gain, so every start is describable by three numbers --
the form issue #154 asks for.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

__all__ = [
    "GAIN_ORDER",
    "INIT_SCHEMES",
    "LEGACY_UNIFORM_BOX",
    "LEGACY_UNIFORM_SEED",
    "SECVD_BATCH_SIZE",
    "SECVD_INIT_SCHEME",
    "SECVD_INIT_SEED",
    "SECVD_INIT_SPREAD",
    "describe_initial_gains",
    "sample_initial_gains",
]

# Number of independently optimized gain initializations, matching the batch
# dimension of the legacy archives.
SECVD_BATCH_SIZE = 6

SECVD_INIT_SEED = 0
SECVD_INIT_SPREAD = 3.0
SECVD_INIT_SCHEME = "log_uniform_v1"

# Iterated explicitly so that adding a fourth gain later cannot shift the keys
# consumed by the first three and silently resample them.
GAIN_ORDER = ("Kp", "Ki", "Kd")

# The dormant sampling box found in the pre-packaging generator
# (OPTso_closedloop_pcs3at_ws_trajectory_multishoot.py). That branch never
# executed, and the box belongs to a different controller parametrization, so
# this is retained only to test a recovered generator against -- never as the
# recovered value itself.
LEGACY_UNIFORM_BOX = {
    "Kp": (10.0, 60.0),
    "Ki": (5.0, 40.0),
    "Kd": (2.0, 5.0),
}
LEGACY_UNIFORM_SEED = 35

INIT_SCHEMES = ("log_uniform_v1", "legacy_uniform")


def _validate(
    nominal: dict[str, Array], batch_size: int, scheme: str, spread: float
) -> None:
    """Reject an initialization request that cannot produce finite gains.

    Args:
        nominal: Nominal gains keyed by :data:`GAIN_ORDER`.
        batch_size: Number of starts to produce.
        scheme: Name from :data:`INIT_SCHEMES`.
        spread: Multiplicative half-width for ``log_uniform_v1``.

    Raises:
        ValueError: If any argument is outside its supported range.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if scheme not in INIT_SCHEMES:
        raise ValueError(f"Unknown init scheme {scheme!r}; expected {INIT_SCHEMES}")
    if set(nominal) != set(GAIN_ORDER):
        raise ValueError(
            f"Nominal gains must be keyed by {GAIN_ORDER}, got {sorted(nominal)}"
        )
    if scheme == "log_uniform_v1" and not spread > 1.0:
        raise ValueError(f"spread must be greater than 1.0, got {spread}")
    for name in GAIN_ORDER:
        value = jnp.asarray(nominal[name])
        if not bool(jnp.all(jnp.isfinite(value))):
            raise ValueError(f"Nominal {name} must be finite")
        # log_uniform_v1 perturbs multiplicatively, so a zero or negative
        # nominal gain has no well-defined neighbourhood to sample from.
        if scheme == "log_uniform_v1" and not bool(jnp.all(value > 0.0)):
            raise ValueError(
                f"Nominal {name} must be strictly positive for log_uniform_v1"
            )


def sample_initial_gains(
    nominal: dict[str, Array],
    *,
    batch_size: int = SECVD_BATCH_SIZE,
    seed: int = SECVD_INIT_SEED,
    scheme: str = SECVD_INIT_SCHEME,
    spread: float = SECVD_INIT_SPREAD,
) -> dict[str, Array]:
    """Build a batch of gain initializations whose first entry is the nominal set.

    ``log_uniform_v1`` scales the whole nominal vector for gain ``g`` by a factor
    drawn log-uniformly from ``[1 / spread, spread]``, which treats gains as the
    scale parameters they are and preserves any anisotropy already present in the
    nominal. ``legacy_uniform`` instead draws an absolute value from
    :data:`LEGACY_UNIFORM_BOX` and applies it to every component.

    Args:
        nominal: Nominal gains keyed by :data:`GAIN_ORDER`, each of shape ``(m,)``.
        batch_size: Number of starts ``B``. ``1`` returns the nominal alone.
        seed: PRNG seed. Recorded in the archive alongside the scheme.
        scheme: Name from :data:`INIT_SCHEMES`.
        spread: Multiplicative half-width for ``log_uniform_v1``; unused otherwise.

    Returns:
        Gains keyed by :data:`GAIN_ORDER`, each of shape ``(B, m)``, with entry
        ``0`` equal to the corresponding nominal value.

    Raises:
        ValueError: If the request cannot produce finite gains.
    """
    _validate(nominal, batch_size, scheme, spread)

    key = jax.random.PRNGKey(seed)
    gains: dict[str, Array] = {}
    for name in GAIN_ORDER:
        key, subkey = jax.random.split(key)
        value = jnp.asarray(nominal[name])
        if batch_size == 1:
            gains[name] = value[None, ...]
            continue

        # One scalar per start, broadcast across actuators, so that each start
        # is fully described by three numbers.
        factor_shape = (batch_size - 1,) + (1,) * value.ndim
        if scheme == "log_uniform_v1":
            half_width = jnp.log10(spread)
            factor = 10.0 ** jax.random.uniform(
                subkey,
                factor_shape,
                dtype=value.dtype,
                minval=-half_width,
                maxval=half_width,
            )
            sampled = value[None, ...] * factor
        else:
            low, high = LEGACY_UNIFORM_BOX[name]
            sampled = jnp.ones_like(value)[None, ...] * jax.random.uniform(
                subkey, factor_shape, dtype=value.dtype, minval=low, maxval=high
            )
        gains[name] = jnp.concatenate([value[None, ...], sampled], axis=0)

    return gains


def describe_initial_gains(gains: dict[str, Array]) -> str:
    """Render the batch of initializations as the table issue #154 asks for.

    Args:
        gains: Output of :func:`sample_initial_gains`.

    Returns:
        A table with one row per start, showing each gain's first component.
    """
    batch_size = int(jnp.asarray(gains[GAIN_ORDER[0]]).shape[0])
    header = "  start  " + "".join(f"{name:>14s}" for name in GAIN_ORDER)
    rows = [header, "  " + "-" * (len(header) - 2)]
    for index in range(batch_size):
        cells = "".join(
            f"{float(jnp.asarray(gains[name])[index].reshape(-1)[0]):14.4f}"
            for name in GAIN_ORDER
        )
        label = f"{index:>7d}" + (" *" if index == 0 else "  ")
        rows.append(label + cells)
    rows.append("  * nominal")
    return "\n".join(rows)
