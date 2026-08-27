"""Tests for user-visible Warp launch configuration."""

from __future__ import annotations

import pytest

from soromox.systems.execution import (
    DEFAULT_PCS_BLOCK_DIM,
    DEFAULT_PLANAR_PCS_BLOCK_DIM,
    validate_block_dim,
)
from soromox.systems.execution.warp.config import gvs_block_dim


@pytest.mark.parametrize("value", [32, 64, 128, 192, 1024])
def test_validate_block_dim_accepts_cuda_warp_multiples(value: int) -> None:
    """Accept every legal user override from one warp to one CUDA block."""

    assert validate_block_dim(value) == value


@pytest.mark.parametrize("value", [True, 32.0, "128", None])
def test_validate_block_dim_rejects_non_integer_values(value: object) -> None:
    """Avoid silently converting ambiguous launch configuration values."""

    with pytest.raises(TypeError, match="integer multiple of 32"):
        validate_block_dim(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-32, 0, 1, 31, 33, 1056])
def test_validate_block_dim_rejects_invalid_integer_values(value: int) -> None:
    """Enforce CUDA's block limit and full-warp granularity."""

    with pytest.raises(ValueError, match="multiple of 32 between 32 and 1024"):
        validate_block_dim(value)


def test_retained_defaults_do_not_depend_on_runtime_gpu_identity() -> None:
    """Keep PCS defaults deterministic and leave tuning to explicit overrides."""

    assert DEFAULT_PLANAR_PCS_BLOCK_DIM == 128
    assert DEFAULT_PCS_BLOCK_DIM == 192


@pytest.mark.parametrize(
    ("num_dofs", "expected"),
    [(1, 128), (64, 128), (65, 192), (256, 192)],
)
def test_gvs_retains_bounded_shape_rule(num_dofs: int, expected: int) -> None:
    """Document the currently benchmarked shape-generic GVS launch policy."""

    assert gvs_block_dim(num_dofs, gpu=True) == expected
    assert gvs_block_dim(num_dofs, gpu=False) == 1
