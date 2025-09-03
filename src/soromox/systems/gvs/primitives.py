__all__ = ["Basis", "Joint", "Link"]
import jax
import jax.numpy as jnp
from jax import Array
from typing import Callable, List, Optional, Tuple


from soromox.systems.gvs.operands import *
from soromox.systems.gvs.strain_bases import (
    dof_Monomial,
    dof_LegendrePolynomial,
    dof_Chebychev,
    dof_Fourier,
    dof_Gaussian,
    dof_IMQ,
)


class Basis:
    BASISTYPE_MAP = {
        "Monomial": 0,
        "Legendre": 1,
        "Chebychev": 2,
        "Fourier": 3,
        "Gaussian": 4,
        "IMQ": 5,
    }

    DOF_BRANCHES = [
        lambda operand: dof_Monomial(operand[0], operand[1]),
        lambda operand: dof_LegendrePolynomial(operand[0], operand[1]),
        lambda operand: dof_Chebychev(operand[0], operand[1]),
        lambda operand: dof_Fourier(operand[0], operand[1]),
        lambda operand: dof_Gaussian(operand[0], operand[1]),
        lambda operand: dof_IMQ(operand[0], operand[1]),
    ]

    @staticmethod
    def make_B_branch(B_fn, max_dof):
        def padded_branch(operand):
            Xs, Bdof, Bodr = operand

            def apply_fn(x):
                return B_fn(x, Bdof, Bodr, max_dof)

            return jax.vmap(apply_fn)(Xs)

        return padded_branch


class Joint:
    JOINTTYPE_MAP = {
        "Revolute": 0,  # Revolute joint
        "Prismatic": 1,  # Prismatic joint
        "Helical": 2,  # Helical joint
        "Cylindrical": 3,  # Cylindrical joint
        "Planar": 4,  # Planar joint
        "Spherical": 5,  # Spherical joint
        "Free": 6,  # Free motion joint
        "Fixed": 7,  # No motion joint
    }
    AXIS_MAP = {
        "x": 0,  # x-axis
        "y": 1,  # y-axis
        "z": 2,  # z-axis
    }
    PLANE_MAP = {
        "xy": 0,  # xy-plane
        "yz": 1,  # yz-plane
        "xz": 2,  # xz-plane
    }
    DICT_JOINT_TYPE_DOF = {
        "Revolute": 1,
        "Prismatic": 1,
        "Helical": 1,
        "Cylindrical": 2,
        "Planar": 3,
        "Spherical": 3,
        "Free": 6,
        "Fixed": 0,
    }

    @staticmethod
    def make_B_branch(
        B_fn: Callable[[JointOperand], Array], dof_joint: int, max_dof: int
    ) -> Callable:
        def padded_branch(operand):
            B_unpadded = B_fn(operand)

            # Case 1: truncate if dof_joint > max_dof
            if dof_joint > max_dof:
                return B_unpadded[:, :max_dof]

            # Case 2: pad if dof_joint < max_dof
            return jnp.pad(
                B_unpadded, ((0, 0), (0, max_dof - dof_joint)), constant_values=0.0
            )

        return padded_branch


class Link:
    section_idx: int
    SECTION_MAP = {
        "Circular": 0,  # Circular cross-section
        "Rectangular": 1,  # Rectangular cross-section
        "Elliptical": 2,  # Elliptical cross-section
    }

    @staticmethod
    def geometric_branches():
        """
        Returns a list of functions that compute the geometric parameters for different section types.
        Each function takes the same operand structure and returns the computed parameters.
        """
        return [
            Link.compute_circular_params,
            Link.compute_rectangular_params,
            Link.compute_elliptical_params,
        ]

    E: Array  # Young's Modulus [N/m²]
    nu: Array  # Poisson Ratio [-1, 0.5]
    G: Array  # Shear modulus [N/m²]
    rho: Array  # Density [kg/m³]
    eta: Array  # Material Damping [N·s/m]
    l: Array  # Length of each divisions of the link (soft link) [m]

    r: Tuple[float, float]  # Initial and final value of the geometrical parameter
    h: Tuple[float, float]  # Initial and final value of the geometrical parameter
    w: Tuple[float, float]  # Initial and final value of the geometrical parameter
    a: Tuple[float, float]  # Initial and final value of the geometrical parameter
    b: Tuple[float, float]  # Initial and final value of the geometrical parameter

    @staticmethod
    def interpolate_param(x, a, b):
        return a + (b - a) * x

    @staticmethod
    def compute_rectangular_params(
        operand: GeometricOperand,
    ) -> Tuple[Array, Array, Array, Array]:
        Xs = operand.Xs
        h_params = operand.h_params
        w_params = operand.w_params

        h_nGauss = Link.interpolate_param(Xs, *h_params)
        w_nGauss = Link.interpolate_param(Xs, *w_params)

        Iy_p = (1 / 12) * h_nGauss * w_nGauss**3
        Iz_p = (1 / 12) * w_nGauss * h_nGauss**3
        Ix_p = Iy_p + Iz_p
        A_p = h_nGauss * w_nGauss
        return Ix_p, Iy_p, Iz_p, A_p

    @staticmethod
    def compute_circular_params(
        operand: GeometricOperand,
    ) -> Tuple[Array, Array, Array, Array]:
        Xs = operand.Xs
        r_params = operand.r_params

        r_nGauss = Link.interpolate_param(Xs, *r_params)

        Iy_p = jnp.pi / 4 * r_nGauss**4
        Iz_p = Iy_p
        Ix_p = Iy_p + Iz_p
        A_p = jnp.pi * r_nGauss**2
        return Ix_p, Iy_p, Iz_p, A_p

    @staticmethod
    def compute_elliptical_params(
        operand: GeometricOperand,
    ) -> Tuple[Array, Array, Array, Array]:
        Xs = operand.Xs
        a_params = operand.a_params
        b_params = operand.b_params

        a_nGauss = Link.interpolate_param(Xs, *a_params)
        b_nGauss = Link.interpolate_param(Xs, *b_params)

        Iy_p = jnp.pi / 4 * a_nGauss * b_nGauss**3
        Iz_p = jnp.pi / 4 * b_nGauss * a_nGauss**3
        Ix_p = Iy_p + Iz_p
        A_p = jnp.pi * a_nGauss * b_nGauss
        return Ix_p, Iy_p, Iz_p, A_p
    