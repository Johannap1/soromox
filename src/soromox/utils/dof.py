"""Utilities for selecting and embedding active degrees of freedom."""

from jax import Array
from jax import numpy as jnp

__all__ = ["build_active_dof_basis"]


def build_active_dof_basis(active_selector: Array) -> Array:
    """Build the canonical basis embedding active DOFs into full coordinates.

    Args:
        active_selector: Boolean array selecting active coordinates in the
            full coordinate layout.

    Returns:
        A basis matrix of shape ``(n_full_coordinates, n_active_dofs)``.
    """
    active_selector = jnp.asarray(active_selector, dtype=bool)
    n_active = int(active_selector.sum().item())
    if n_active == 0:
        return jnp.zeros((active_selector.shape[0], 0))

    active_column = jnp.cumsum(active_selector.astype(jnp.int32)) - 1
    basis = jnp.eye(n_active)[active_column]
    return basis * active_selector[:, None]
