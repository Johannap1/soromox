__all__ = ["PlanarHSAStructure"]

import equinox as eqx
from jax import Array


class PlanarHSAStructure(eqx.Module):
    """Static numerical layout choices for planar HSA."""

    num_gauss_points: int = eqx.field(static=True, default=5)
    strain_selector: Array | None = None
    consider_underactuation: bool = eqx.field(static=True, default=True)
    consider_hysteresis: bool = eqx.field(static=True, default=False)
    eps: float = eqx.field(static=True, default=1e-6)
