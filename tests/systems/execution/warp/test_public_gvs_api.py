"""Public-surface tests for reusable Warp-native GVS mechanics."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys

import pytest


def test_gvs_public_names_are_discoverable_without_loading_warp() -> None:
    """Advertise stable symbols while preserving the optional dependency."""

    from soromox.systems.execution.warp import gvs

    expected = {
        "GVSOperandSource",
        "GVSKinematicsOperands",
        "GVSKinematicsShapes",
        "GVSOperands",
        "GVSPipelineShapes",
        "cell_terms_kernel",
        "cooperative_joint_terms_kernel",
        "gvs_cooperative_node_states_kernel",
        "joint_terms_kernel",
        "gvs_jacobian_samples_kernel",
        "gvs_node_poses_kernel",
        "gvs_node_states_kernel",
        "gvs_pose_samples_kernel",
        "gvs_samples_kernel",
        "launch_cell_terms",
        "launch_cooperative_joint_terms",
        "launch_joint_terms",
        "launch_forward_kinematics",
        "launch_forward_kinematics_and_jacobians",
        "launch_gvs_forward_kinematics",
        "launch_gvs_inertial_jacobians",
        "launch_gvs_kinematics",
        "launch_inertial_jacobians",
        "launch_persistent_chain",
        "persistent_chain_kernel",
    }
    assert expected == set(gvs.__all__)
    assert expected <= set(dir(gvs))


def test_public_gvs_launch_functions_have_documented_warp_contracts() -> None:
    """Expose direct preallocated-buffer launchers for external integrators."""

    pytest.importorskip("warp")
    from soromox.systems.execution.warp.gvs import (
        cell_terms_kernel,
        cooperative_joint_terms_kernel,
        gvs_cooperative_node_states_kernel,
        gvs_jacobian_samples_kernel,
        gvs_node_poses_kernel,
        gvs_node_states_kernel,
        gvs_pose_samples_kernel,
        gvs_samples_kernel,
        joint_terms_kernel,
        launch_cell_terms,
        launch_cooperative_joint_terms,
        launch_forward_kinematics,
        launch_forward_kinematics_and_jacobians,
        launch_gvs_forward_kinematics,
        launch_gvs_inertial_jacobians,
        launch_gvs_kinematics,
        launch_inertial_jacobians,
        launch_joint_terms,
        launch_persistent_chain,
        persistent_chain_kernel,
    )

    launchers = (
        launch_forward_kinematics,
        launch_forward_kinematics_and_jacobians,
        launch_gvs_forward_kinematics,
        launch_gvs_inertial_jacobians,
        launch_gvs_kinematics,
        launch_inertial_jacobians,
        launch_joint_terms,
        launch_cooperative_joint_terms,
        launch_cell_terms,
        launch_persistent_chain,
    )
    kernels = (
        gvs_cooperative_node_states_kernel,
        gvs_jacobian_samples_kernel,
        gvs_node_poses_kernel,
        gvs_node_states_kernel,
        gvs_pose_samples_kernel,
        gvs_samples_kernel,
        joint_terms_kernel,
        cooperative_joint_terms_kernel,
        cell_terms_kernel,
        persistent_chain_kernel,
    )

    for launcher in launchers:
        documentation = inspect.getdoc(launcher)
        assert documentation is not None
        assert "Args:" in documentation
    specialized = (
        launch_gvs_forward_kinematics,
        launch_gvs_inertial_jacobians,
        launch_gvs_kinematics,
        launch_joint_terms,
        launch_cooperative_joint_terms,
        launch_cell_terms,
        launch_persistent_chain,
    )
    for launcher in specialized:
        assert len(inspect.signature(launcher).parameters) >= 10
    for kernel in kernels:
        assert kernel is not None


def test_operand_contract_does_not_eagerly_load_kernel_modules() -> None:
    """Let orchestration inspect model operands without importing Warp kernels."""

    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from soromox.systems.execution.warp.gvs import GVSOperands; "
                "assert GVSOperands is not None; "
                "assert 'soromox.systems.execution.warp.gvs.chain' "
                "not in sys.modules; "
                "assert 'soromox.systems.execution.warp.gvs.cell' "
                "not in sys.modules; "
                "assert 'soromox.systems.execution.warp.gvs.joint' "
                "not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert result.returncode == 0, result.stderr
