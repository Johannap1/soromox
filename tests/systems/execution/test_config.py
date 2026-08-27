"""Tests for public execution-backend configuration."""

from __future__ import annotations

import pytest

from soromox.systems.execution import (
    DEFAULT_PCS_BLOCK_DIM,
    DEFAULT_PLANAR_PCS_BLOCK_DIM,
    PCSBackendParams,
)


@pytest.mark.parametrize("value", [32, 64, 128, 192, 1024])
def test_pcs_backend_params_accept_cuda_warp_multiples(value: int) -> None:
    """Accept every legal override from one warp to one CUDA block."""

    assert PCSBackendParams(warp_block_dim=value).warp_block_dim == value


@pytest.mark.parametrize("value", [True, 32.0, "128", None])
def test_pcs_backend_params_reject_non_integer_values(value: object) -> None:
    """Avoid silently converting ambiguous launch configuration values."""

    with pytest.raises(TypeError, match="integer multiple of 32"):
        PCSBackendParams(warp_block_dim=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-32, 0, 1, 31, 33, 1056])
def test_pcs_backend_params_reject_invalid_integer_values(value: int) -> None:
    """Enforce CUDA's block limit and full-warp granularity."""

    with pytest.raises(ValueError, match="multiple of 32 between 32 and 1024"):
        PCSBackendParams(warp_block_dim=value)


def test_pcs_defaults_do_not_depend_on_runtime_gpu_identity() -> None:
    """Keep defaults deterministic and leave tuning to explicit parameters."""

    assert DEFAULT_PLANAR_PCS_BLOCK_DIM == 128
    assert DEFAULT_PCS_BLOCK_DIM == 192
