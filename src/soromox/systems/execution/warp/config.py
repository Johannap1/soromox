"""Static launch configuration shared by Warp dynamics executors."""

from __future__ import annotations

DEFAULT_PLANAR_PCS_BLOCK_DIM = 128
DEFAULT_PCS_BLOCK_DIM = 192


def validate_block_dim(value: int) -> int:
    """Validate a user-visible CUDA thread-block dimension.

    Args:
        value: Requested threads per block. The value must contain whole CUDA
            warps and remain within CUDA's portable per-block limit.

    Returns:
        ``value`` unchanged after validation.

    Raises:
        TypeError: If ``value`` is not an integer or is a boolean.
        ValueError: If ``value`` is outside ``[32, 1024]`` or is not divisible
            by 32.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "warp_block_dim must be an integer multiple of 32, "
            f"got {type(value).__name__}."
        )
    if value < 32 or value > 1024 or value % 32 != 0:
        raise ValueError(
            f"warp_block_dim must be a multiple of 32 between 32 and 1024, got {value}."
        )
    return value


def gvs_block_dim(num_dofs: int, *, gpu: bool) -> int:
    """Return the retained shape-generic GVS launch configuration.

    Args:
        num_dofs: Number of active generalized coordinates.
        gpu: Whether execution targets a GPU. CPU execution uses one lane.

    Returns:
        One lane for CPU execution, 128 threads for at most 64 active
        coordinates, and 192 threads for larger systems.
    """

    if not gpu:
        return 1
    return 192 if num_dofs > 64 else 128


__all__ = [
    "DEFAULT_PCS_BLOCK_DIM",
    "DEFAULT_PLANAR_PCS_BLOCK_DIM",
    "gvs_block_dim",
    "validate_block_dim",
]
