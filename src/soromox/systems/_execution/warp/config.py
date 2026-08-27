"""Static launch configuration shared by Warp dynamics executors."""

from __future__ import annotations

DEFAULT_PLANAR_PCS_BLOCK_DIM = 128
DEFAULT_PCS_BLOCK_DIM = 192


def validate_block_dim(value: int) -> int:
    """Validate a CUDA thread-block dimension exposed by a system model."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "warp_block_dim must be an integer multiple of 32, "
            f"got {type(value).__name__}."
        )
    if value < 32 or value > 1024 or value % 32 != 0:
        raise ValueError(
            "warp_block_dim must be a multiple of 32 between 32 and 1024, "
            f"got {value}."
        )
    return value


def gvs_block_dim(num_dofs: int, *, gpu: bool) -> int:
    """Return the retained shape-generic GVS launch configuration."""

    if not gpu:
        return 1
    return 192 if num_dofs > 64 else 128


__all__ = [
    "DEFAULT_PCS_BLOCK_DIM",
    "DEFAULT_PLANAR_PCS_BLOCK_DIM",
    "gvs_block_dim",
    "validate_block_dim",
]
