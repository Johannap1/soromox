__all__ = ["LinkAttributes", "JointAttributes", "BasisAttributes"]
from dataclasses import dataclass, field
import jax
from jax import Array
from typing import List, Literal, Optional, Tuple, Union


@dataclass
class LinkAttributes:
    section: Literal["Circular", "Rectangular", "Elliptical"]
    E: float
    nu: float
    rho: float
    eta: float
    L: float

    r_i: Optional[float] = 0.0
    r_f: Optional[float] = 0.0
    h_i: Optional[float] = 0.0
    h_f: Optional[float] = 0.0
    w_i: Optional[float] = 0.0
    w_f: Optional[float] = 0.0
    a_i: Optional[float] = 0.0
    a_f: Optional[float] = 0.0
    b_i: Optional[float] = 0.0
    b_f: Optional[float] = 0.0


@dataclass
class JointAttributes:
    jointtype: Literal[
        "Revolute",
        "Prismatic",
        "Helical",
        "Cylindrical",
        "Planar",
        "Spherical",
        "Free",
        "Fixed",
    ]

    axis: Literal["x", "y", "z"] = "x"
    plane: Literal["xy", "yz", "xz"] = "xy"
    pitch: float = 0.0

    K_joint: Union[Array, List] = field(default_factory=list)


@dataclass
class BasisAttributes:
    basistype: Literal[
        "Monomial", "Legendre", "Chebychev", "Fourier", "Gaussian", "IMQ"
    ]
    Bdof: Union[
        Array, List
    ]  # shape (6,1) indicating whether each type of deformation is selected (1) or not (0)
    Bodr: Union[
        Array, List
    ]  # shape (6,1) indicating the orders of the basis functions for each type of deformation
    xi_ref: Union[
        Array, List
    ]  # shape (6,1) indicating the reference strain values for each type of deformation