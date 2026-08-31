"""User-facing configuration for optional system execution backends."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_PLANAR_PCS_BLOCK_DIM = 128
DEFAULT_PCS_BLOCK_DIM = 192


def _validate_warp_block_dim(value: int) -> int:
    """Validate a CUDA thread-block dimension used by PCS Warp kernels.

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


@dataclass(frozen=True)
class PCSBackendParams:
    """Tune execution of the optional PlanarPCS and PCS Warp kernels.

    Backend parameters are intentionally separate from the ``backend`` choice:
    ``backend`` selects ``"auto"``, ``"jax"``, or ``"warp"``, whereas this
    object configures the selected PCS implementation. Keeping the options in
    one typed object allows future tuning parameters to be added without
    expanding the system constructors with backend-specific keywords.

    Attributes:
        warp_block_dim: CUDA threads per persistent dynamics block. It must be
            a multiple of 32 between 32 and 1024. The recommended defaults are
            128 for :class:`~soromox.systems.pcs.PlanarPCS` and 192 for
            :class:`~soromox.systems.pcs.PCS`.
    """

    warp_block_dim: int

    def __post_init__(self) -> None:
        """Validate the immutable launch configuration.

        Returns:
            None.

        Raises:
            TypeError: If :attr:`warp_block_dim` is not an integer.
            ValueError: If :attr:`warp_block_dim` is not a supported CUDA block
                dimension.
        """

        _validate_warp_block_dim(self.warp_block_dim)


__all__ = [
    "DEFAULT_PCS_BLOCK_DIM",
    "DEFAULT_PLANAR_PCS_BLOCK_DIM",
    "PCSBackendParams",
]
