"""Public-surface tests for reusable Warp-native PCS mechanics."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys

import pytest


def test_pcs_public_names_are_discoverable_without_loading_kernels() -> None:
    """Advertise stable planar and spatial PCS integration symbols lazily."""

    from soromox.systems.execution.warp import pcs

    expected = {
        "PCSOperandSource",
        "PCSOperands",
        "PCSPipelineShapes",
        "launch_planar_local_operators",
        "launch_planar_persistent_chain",
        "launch_spatial_local_operators",
        "launch_spatial_persistent_chain",
        "planar_local_operators_kernel",
        "planar_persistent_chain_kernel",
        "spatial_local_operators_kernel",
        "spatial_persistent_chain_kernel",
    }
    assert expected == set(pcs.__all__)
    assert expected <= set(dir(pcs))


def test_public_pcs_launch_functions_have_documented_warp_contracts() -> None:
    """Expose preallocated-buffer launchers for both PCS dimensions."""

    pytest.importorskip("warp")
    from soromox.systems.execution.warp.pcs import (
        launch_planar_local_operators,
        launch_planar_persistent_chain,
        launch_spatial_local_operators,
        launch_spatial_persistent_chain,
        planar_local_operators_kernel,
        planar_persistent_chain_kernel,
        spatial_local_operators_kernel,
        spatial_persistent_chain_kernel,
    )

    launchers = (
        launch_planar_local_operators,
        launch_planar_persistent_chain,
        launch_spatial_local_operators,
        launch_spatial_persistent_chain,
    )
    kernels = (
        planar_local_operators_kernel,
        planar_persistent_chain_kernel,
        spatial_local_operators_kernel,
        spatial_persistent_chain_kernel,
    )
    for launcher in launchers:
        documentation = inspect.getdoc(launcher)
        assert documentation is not None
        assert "Args:" in documentation
        assert len(inspect.signature(launcher).parameters) >= 10
    for kernel in kernels:
        assert kernel is not None


def test_pcs_operand_contract_does_not_eagerly_load_kernel_modules() -> None:
    """Inspect PCS operands without importing optional dimension kernels."""

    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from soromox.systems.execution.warp.pcs import PCSOperands; "
                "assert PCSOperands is not None; "
                "assert 'soromox.systems.execution.warp.pcs.planar_kernels' "
                "not in sys.modules; "
                "assert 'soromox.systems.execution.warp.pcs.spatial_kernels' "
                "not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert result.returncode == 0, result.stderr
