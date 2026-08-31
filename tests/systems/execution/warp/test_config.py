"""Tests for user-visible Warp launch configuration."""

from __future__ import annotations

import pytest

from soromox.systems.execution.warp.config import gvs_block_dim


@pytest.mark.parametrize(
    ("num_dofs", "expected"),
    [(1, 128), (64, 128), (65, 192), (256, 192)],
)
def test_gvs_retains_bounded_shape_rule(num_dofs: int, expected: int) -> None:
    """Document the currently benchmarked shape-generic GVS launch policy."""

    assert gvs_block_dim(num_dofs, gpu=True) == expected
    assert gvs_block_dim(num_dofs, gpu=False) == 1
