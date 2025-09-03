__all__ = [
    "JointOperand",
    "GeometricOperand",
]

from dataclasses import dataclass, field
import jax
from jax import Array
from typing import List, Literal, Optional, Tuple, Union


@jax.tree_util.register_pytree_node_class
@dataclass
class JointOperand:
    axis_idx: int
    pitch: float
    plane_idx: int

    def tree_flatten(self):
        children = (self.axis_idx, self.pitch, self.plane_idx)
        aux_data = None  # aucun champ statique à exclure ici
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass
class GeometricOperand:
    Xs: Array
    r_params: Tuple[Array, Array]
    h_params: Tuple[Array, Array]
    w_params: Tuple[Array, Array]
    a_params: Tuple[Array, Array]
    b_params: Tuple[Array, Array]

    def tree_flatten(self):
        children = (
            self.Xs,
            self.r_params,
            self.h_params,
            self.w_params,
            self.a_params,
            self.b_params,
        )
        aux_data = None  # aucun champ statique à exclure ici
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)