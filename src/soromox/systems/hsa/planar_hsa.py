from typing import Any

import equinox as eqx
import jax
from jax import Array, lax, vmap
from jax import numpy as jnp

from soromox.systems.hsa.params import PlanarHSAParams
from soromox.systems.hsa.structures import PlanarHSAStructure
from soromox.systems.pcs.planar_pcs import PlanarPCS
from soromox.systems.soft_robot import CrossSectionGeometry, SoftRobot
from soromox.utils.array_math import blk_diag
from soromox.utils.dof import build_active_dof_basis
from soromox.utils.integration import gauss_quadrature

__all__ = ["PlanarHSA"]


class PlanarHSA(PlanarPCS):
    """
    Kinematic and dynamic model for planar Handed Shearing Auxetics (HSA) robots.

    The model uses a piecewise-constant virtual backbone with strain ordering
    ``[kappa_b, sigma_sh, sigma_a]`` per segment: bending curvature, shear
    strain, and axial strain. Physical rods are mapped to the virtual backbone
    through their offsets, and rigid caps, platforms, and the payload are
    included in forward kinematics and dynamics.

    PlanarHSA reuses the planar PCS layout, SE(2) propagation, fixed
    Gauss--Legendre quadrature, strain selection, and differentiable-system
    interfaces. Kinematics, Jacobians, energies, forces, and forward dynamics
    are evaluated numerically with JAX and support JIT compilation and
    autodiff. Optional underactuation maps motor inputs to rod forces, while
    optional Bouc--Wen hysteresis augments the elastic response.

    Based on:
        Stölzle, M., Rus, D., & Della Santina, C. (2024). An experimental study
        of model-based control for planar handed shearing auxetics robots. In
        Experimental Robotics: The 18th International Symposium (pp. 153-167).
        Springer. https://doi.org/10.1007/978-3-031-63596-0_14

    Attributes:
        num_segments: Number of constant-strain segments.
        num_rods_per_segment: Number of physical rods per segment.
        num_dofs: Number of active strain coordinates.
        num_actuators: Number of motor or direct-torque input channels.
        consider_underactuation: Whether motor-to-rod actuation is enabled.
        consider_hysteresis: Whether Bouc--Wen hysteresis is enabled.
        num_hysteresis: Number of hysteresis state variables.
        B_xi: Basis mapping active coordinates to virtual-backbone strains.
        L: Flexible segment lengths.
        L_cum: Cumulative flexible segment lengths.
        length: Total backbone length inherited from :class:`SoftRobot`.
        rod_offset: Physical rod offsets from the virtual backbone.
        platform_dimension: Platform width, height, and depth per segment.
        proximal_cap_length: Rigid proximal cap lengths.
        distal_cap_length: Rigid distal cap lengths.
        end_effector_offset: End-effector pose offset ``[theta, x, y]``.
        bending_reference: Reference bending strains per physical rod.
        shear_reference: Reference shear strains per physical rod.
        axial_reference: Reference axial strains per physical rod.
        phi_max: Motor limits for underactuated operation.
        hysteresis_basis: Basis mapping hysteresis states to full strains.
        hysteresis_alpha: Post-yield to pre-yield stiffness ratios.
        hysteresis_A: Bouc--Wen ``A`` parameters.
        hysteresis_n: Bouc--Wen exponents.
        hysteresis_beta: Bouc--Wen ``beta`` parameters.
        hysteresis_gamma: Bouc--Wen ``gamma`` parameters.

    """

    num_rods_per_segment: int = eqx.field(static=True)
    consider_underactuation: bool = eqx.field(static=True)
    consider_hysteresis: bool = eqx.field(static=True)
    num_hysteresis: int = eqx.field(static=True)

    rod_offset: Array
    platform_dimension: Array
    proximal_cap_length: Array
    distal_cap_length: Array
    end_effector_offset: Array
    phi_max: Array
    bending_reference: Array
    shear_reference: Array
    axial_reference: Array
    strain_coupling: Array
    rod_height: Array
    rod_outer_radius: Array
    rod_inner_radius: Array
    rod_density: Array
    platform_density: Array
    end_cap_density: Array
    nominal_bending_stiffness: Array
    nominal_shear_stiffness: Array
    nominal_axial_stiffness: Array
    bending_shear_stiffness: Array
    bending_stiffness_correction: Array
    shear_stiffness_correction: Array
    axial_stiffness_correction: Array
    bending_damping: Array
    shear_damping: Array
    axial_damping: Array
    platform_mass: Array
    platform_center_of_gravity: Array
    hysteresis_basis: Array
    hysteresis_alpha: Array
    hysteresis_A: Array
    hysteresis_n: Array
    hysteresis_beta: Array
    hysteresis_gamma: Array

    params: PlanarHSAParams

    def __init__(
        self,
        params: PlanarHSAParams,
        structure: PlanarHSAStructure | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(params, PlanarHSAParams):
            raise TypeError("params must be a PlanarHSAParams instance.")
        if structure is None:
            structure = PlanarHSAStructure()
        if not isinstance(structure, PlanarHSAStructure):
            raise TypeError("structure must be a PlanarHSAStructure instance.")
        params.validate()

        # HSA has a distinct physical parameter schema, so initialize the
        # common SoftRobot fields directly instead of fabricating PCS params.
        SoftRobot.__init__(self, eps=structure.eps, base_pose=params.base_pose, **kwargs)
        self.params = params

        n_segments = int(params.length.shape[0])
        n_rods = int(params.rod_offset.shape[1])
        n_strains = 3 * n_segments
        self.num_segments = n_segments
        self.num_strains = n_strains
        self.num_rods_per_segment = n_rods
        self.consider_underactuation = bool(structure.consider_underactuation)
        self.consider_hysteresis = bool(structure.consider_hysteresis)
        self.num_dofs = n_strains
        self.num_actuators = n_segments * n_rods if self.consider_underactuation else n_strains

        self.base_pose = jnp.asarray(params.base_pose, dtype=jnp.float64)
        self.g = jnp.concatenate(
            [jnp.zeros((1,), dtype=jnp.float64), jnp.asarray(params.gravity, dtype=jnp.float64)]
        )
        self.L = jnp.asarray(params.length, dtype=jnp.float64)
        self.L_cum = jnp.cumsum(jnp.concatenate([jnp.zeros((1,)), self.L]))
        self.r = None
        self.rho = None
        self.E = None
        self.G = None
        self.scale_rotational_basis_by_length = False

        num_gauss_points = structure.num_gauss_points
        if not isinstance(num_gauss_points, int) or num_gauss_points < 1:
            raise ValueError("num_gauss_points must be a positive integer.")
        self.num_gauss_points = num_gauss_points
        (
            self.integration_points,
            self.integration_weights,
            self.num_integration_points,
        ) = gauss_quadrature(num_gauss_points)

        selector = structure.strain_selector
        if selector is None:
            selector = jnp.ones((n_strains,), dtype=bool)
        else:
            selector = jnp.asarray(selector)
            if selector.dtype != jnp.bool_ or selector.size != n_strains:
                raise ValueError(
                    f"strain_selector must be a boolean array with {n_strains} elements."
                )
            selector = selector.reshape((n_strains,))
        self.B_xi_unscaled = build_active_dof_basis(selector)
        self.B_xi = self.B_xi_unscaled
        self.num_active_strains = jnp.sum(selector)
        self.num_dofs = int(self.num_active_strains.item())

        self._set_hsa_params(params)
        self.xi_ref = self.ref_strains()
        self._set_hysteresis(params)

        # HSA supplies its own rod material model and therefore has no
        # synthetic PlanarPCS material caches.
        self.M_segments = None
        self.K_full = None
        self.K_active = None
        self.D_full = None
        self.D_active = None
        self.actuators = ()
        self.passive_elements = ()

    @property
    def is_planar(self) -> bool:
        return True

    @property
    def segment_length(self) -> Array:
        return jnp.asarray(self.L)

    def _set_hsa_params(self, params: PlanarHSAParams) -> None:
        self.rod_offset = jnp.asarray(params.rod_offset, dtype=jnp.float64)
        self.platform_dimension = jnp.asarray(params.platform_dimension, dtype=jnp.float64)
        self.proximal_cap_length = jnp.asarray(params.proximal_cap_length, dtype=jnp.float64)
        self.distal_cap_length = jnp.asarray(params.distal_cap_length, dtype=jnp.float64)
        self.end_effector_offset = jnp.asarray(params.end_effector_offset, dtype=jnp.float64)
        self.phi_max = jnp.asarray(params.phi_max, dtype=jnp.float64)
        self.bending_reference = jnp.asarray(params.bending_reference, dtype=jnp.float64)
        self.shear_reference = jnp.asarray(params.shear_reference, dtype=jnp.float64)
        self.axial_reference = jnp.asarray(params.axial_reference, dtype=jnp.float64)
        self.strain_coupling = jnp.asarray(params.strain_coupling, dtype=jnp.float64)
        self.rod_height = jnp.asarray(params.rod_height, dtype=jnp.float64)
        self.rod_outer_radius = jnp.asarray(params.rod_outer_radius, dtype=jnp.float64)
        self.rod_inner_radius = jnp.asarray(params.rod_inner_radius, dtype=jnp.float64)
        self.rod_density = jnp.asarray(params.rod_density, dtype=jnp.float64)
        self.platform_density = jnp.asarray(params.platform_density, dtype=jnp.float64)
        self.end_cap_density = jnp.asarray(params.end_cap_density, dtype=jnp.float64)
        self.nominal_bending_stiffness = jnp.asarray(params.nominal_bending_stiffness, dtype=jnp.float64)
        self.nominal_shear_stiffness = jnp.asarray(params.nominal_shear_stiffness, dtype=jnp.float64)
        self.nominal_axial_stiffness = jnp.asarray(params.nominal_axial_stiffness, dtype=jnp.float64)
        self.bending_shear_stiffness = jnp.asarray(params.bending_shear_stiffness, dtype=jnp.float64)
        self.bending_stiffness_correction = jnp.asarray(params.bending_stiffness_correction, dtype=jnp.float64)
        self.shear_stiffness_correction = jnp.asarray(params.shear_stiffness_correction, dtype=jnp.float64)
        self.axial_stiffness_correction = jnp.asarray(params.axial_stiffness_correction, dtype=jnp.float64)
        self.bending_damping = jnp.asarray(params.bending_damping, dtype=jnp.float64)
        self.shear_damping = jnp.asarray(params.shear_damping, dtype=jnp.float64)
        self.axial_damping = jnp.asarray(params.axial_damping, dtype=jnp.float64)
        self.platform_mass = jnp.asarray(params.platform_mass, dtype=jnp.float64)
        self.platform_center_of_gravity = jnp.asarray(params.platform_center_of_gravity, dtype=jnp.float64)

    def _set_hysteresis(self, params: PlanarHSAParams) -> None:
        if self.consider_hysteresis:
            self.hysteresis_basis = jnp.asarray(params.hysteresis_basis, dtype=jnp.float64)
            if self.hysteresis_basis.ndim != 2 or self.hysteresis_basis.shape[0] != self.num_strains:
                raise ValueError(
                    "hysteresis_basis must have shape (3 * num_segments, num_hysteresis)."
                )
            self.num_hysteresis = int(self.hysteresis_basis.shape[1])
            self.hysteresis_alpha = jnp.asarray(params.hysteresis_alpha, dtype=jnp.float64)
            self.hysteresis_A = jnp.asarray(params.hysteresis_A, dtype=jnp.float64)
            self.hysteresis_n = jnp.asarray(params.hysteresis_n, dtype=jnp.float64)
            self.hysteresis_beta = jnp.asarray(params.hysteresis_beta, dtype=jnp.float64)
            self.hysteresis_gamma = jnp.asarray(params.hysteresis_gamma, dtype=jnp.float64)
        else:
            self.num_hysteresis = 0
            self.hysteresis_basis = jnp.zeros((self.num_strains, 0), dtype=jnp.float64)
            self.hysteresis_alpha = jnp.zeros((self.num_dofs,), dtype=jnp.float64)
            self.hysteresis_A = jnp.zeros((1,), dtype=jnp.float64)
            self.hysteresis_n = jnp.zeros((1,), dtype=jnp.float64)
            self.hysteresis_beta = jnp.zeros((1,), dtype=jnp.float64)
            self.hysteresis_gamma = jnp.zeros((1,), dtype=jnp.float64)

    def with_params(self, params: PlanarHSAParams) -> "PlanarHSA":
        if not isinstance(params, PlanarHSAParams):
            raise TypeError("params must be a PlanarHSAParams instance.")
        params.validate()
        if params.length.shape != self.L.shape:
            raise ValueError("length shape changes the model structure; construct a new PlanarHSA.")
        if params.rod_offset.shape != self.rod_offset.shape:
            raise ValueError("rod_offset shape changes the model structure; construct a new PlanarHSA.")

        values = {
            "params": params,
            "base_pose": jnp.asarray(params.base_pose, dtype=jnp.float64),
            "g": jnp.concatenate(
                [jnp.zeros((1,), dtype=jnp.float64), jnp.asarray(params.gravity, dtype=jnp.float64)]
            ),
            "L": jnp.asarray(params.length, dtype=jnp.float64),
            "L_cum": jnp.cumsum(jnp.concatenate([jnp.zeros((1,)), params.length])),
        }
        updated = self
        for name, value in values.items():
            updated = eqx.tree_at(
                lambda model, field=name: getattr(model, field), updated, value
            )
        for name in (
            "rod_offset",
            "platform_dimension",
            "proximal_cap_length",
            "distal_cap_length",
            "end_effector_offset",
            "phi_max",
            "bending_reference",
            "shear_reference",
            "axial_reference",
            "strain_coupling",
            "rod_height",
            "rod_outer_radius",
            "rod_inner_radius",
            "rod_density",
            "platform_density",
            "end_cap_density",
            "nominal_bending_stiffness",
            "nominal_shear_stiffness",
            "nominal_axial_stiffness",
            "bending_shear_stiffness",
            "bending_stiffness_correction",
            "shear_stiffness_correction",
            "axial_stiffness_correction",
            "bending_damping",
            "shear_damping",
            "axial_damping",
            "platform_mass",
            "platform_center_of_gravity",
        ):
            value = getattr(params, name)
            updated = eqx.tree_at(
                lambda model, field=name: getattr(model, field),
                updated,
                jnp.asarray(value, dtype=jnp.float64),
            )
        updated = eqx.tree_at(
            lambda model: model.xi_ref, updated, updated.ref_strains()
        )
        if self.consider_hysteresis:
            updated = eqx.tree_at(
                lambda model: (
                    model.hysteresis_basis,
                    model.hysteresis_alpha,
                    model.hysteresis_A,
                    model.hysteresis_n,
                    model.hysteresis_beta,
                    model.hysteresis_gamma,
                ),
                updated,
                (
                    jnp.asarray(params.hysteresis_basis, dtype=jnp.float64),
                    jnp.asarray(params.hysteresis_alpha, dtype=jnp.float64),
                    jnp.asarray(params.hysteresis_A, dtype=jnp.float64),
                    jnp.asarray(params.hysteresis_n, dtype=jnp.float64),
                    jnp.asarray(params.hysteresis_beta, dtype=jnp.float64),
                    jnp.asarray(params.hysteresis_gamma, dtype=jnp.float64),
                ),
            )
        return updated

    def update_params(self, **updates: Array) -> "PlanarHSA":
        return self.with_params(self.params.replace(**updates))

    def cross_section_geometry(self, q: Array, s: Array) -> tuple[Array, Array]:
        del q
        segment_idx, _ = self.classify_segment(s)
        radius = jnp.max(jnp.abs(self.rod_offset[segment_idx]))
        return (
            jnp.asarray(CrossSectionGeometry.CIRCULAR, dtype=jnp.int32),
            jnp.array([radius]),
        )

    def _reference_physical_strains(self) -> Array:
        return jnp.stack(
            [self.bending_reference, self.shear_reference, self.axial_reference],
            axis=-1,
        )

    def beta(self, vxi: Array) -> Array:
        vxi = jnp.asarray(vxi).reshape(self.num_segments, 3)
        physical = jnp.broadcast_to(
            vxi[:, None, :], (self.num_segments, self.num_rods_per_segment, 3)
        )
        return physical.at[:, :, 2].set(
            physical[:, :, 2] + self.rod_offset * physical[:, :, 0]
        )

    def beta_inv(self, pxi: Array) -> Array:
        pxi = jnp.asarray(pxi).reshape(self.num_segments, self.num_rods_per_segment, 3)
        virtual = jnp.mean(pxi, axis=1)
        virtual = virtual.at[:, 2].set(
            virtual[:, 2] - jnp.mean(self.rod_offset * pxi[:, :, 0], axis=1)
        )
        return virtual.reshape(self.num_strains)

    def ref_strains(self) -> Array:
        return self.beta_inv(self._reference_physical_strains())

    def strain(self, q: Array) -> Array:
        return self.B_xi @ q + self.ref_strains()

    def apply_eps_to_bend_strains(self, xi: Array, eps: float | None = None) -> Array:
        if eps is None:
            eps = self.global_eps
        xi_rows = jnp.asarray(xi).reshape((-1, 3))
        sign = jnp.sign(xi_rows[:, 0])
        sign = jnp.where(sign == 0, 1.0, sign)
        bending = jnp.where(jnp.abs(xi_rows[:, 0]) < eps, sign * eps, xi_rows[:, 0])
        return jnp.stack((bending, xi_rows[:, 1], xi_rows[:, 2]), axis=1).reshape(-1)

    def _relative_pose(self, xi_i: Array, arc_length: Array, eps: Array) -> Array:
        del eps
        # The planar SE(2) operator is agnostic to the names of its two linear
        # components. Passing HSA's [kappa, shear, axial] ordering directly
        # reproduces its historical tangent convention while reusing the
        # numerically stable PCS propagator.
        pcs_xi = xi_i
        relative = PlanarPCS._pcs_relative_pose(self, pcs_xi, arc_length)
        return jnp.array(
            [
                jnp.arctan2(relative[1, 0], relative[0, 0]),
                relative[0, 2],
                relative[1, 2],
            ]
        )

    @staticmethod
    def _compose_pose(chi: Array, relative: Array) -> Array:
        theta = chi[0]
        c, s = jnp.cos(theta), jnp.sin(theta)
        p = chi[1:] + jnp.array(
            [c * relative[1] - s * relative[2], s * relative[1] + c * relative[2]]
        )
        return jnp.array([theta + relative[0], p[0], p[1]])

    def _segment_base(self, chi_prev: Array, i: int | Array) -> Array:
        return jnp.array(
            [chi_prev[0], chi_prev[1], chi_prev[2] + self.proximal_cap_length[i]],
            dtype=chi_prev.dtype,
        )

    def _forward_backbone_from_xi(self, xi: Array, s: Array) -> Array:
        segment_idx, s_local = self.classify_segment(s)
        eps = jnp.asarray(self.global_eps, dtype=xi.dtype)
        xi_rows = xi.reshape(self.num_segments, 3)
        chi0 = jnp.asarray(self.base_pose, dtype=xi.dtype)
        zero = jnp.zeros_like(chi0)

        def step(carry: tuple[Array, Array], i: Array) -> tuple[tuple[Array, Array], None]:
            chi_prev, chi_target = carry
            xi_i = xi_rows[i]
            chi_base = self._segment_base(chi_prev, i)
            arc = jnp.where(i == segment_idx, s_local, self.L[i])
            chi_at_arc = self._compose_pose(
                chi_base, self._relative_pose(xi_i, arc, eps)
            )
            chi_target = jnp.where(i == segment_idx, chi_at_arc, chi_target)

            chi_tip = self._compose_pose(
                chi_base, self._relative_pose(xi_i, self.L[i], eps)
            )
            theta = chi_tip[0]
            c, sn = jnp.cos(theta), jnp.sin(theta)
            cap_offset = self.distal_cap_length[i] + self.platform_dimension[i, 1]
            next_p = chi_tip[1:] + jnp.array([-sn * cap_offset, c * cap_offset])
            return (jnp.array([theta, next_p[0], next_p[1]]), chi_target), None

        (_, chi_target), _ = lax.scan(
            step,
            (chi0, zero),
            jnp.arange(self.num_segments, dtype=jnp.int32),
        )
        return chi_target

    def _forward_rod_from_xi(self, xi: Array, s: Array, rod_idx: Array) -> Array:
        segment_idx, _ = self.classify_segment(s)
        chi = self._forward_backbone_from_xi(xi, s)
        offset = self.rod_offset[segment_idx, rod_idx]
        c, sn = jnp.cos(chi[0]), jnp.sin(chi[0])
        p = chi[1:] + jnp.array([c * offset, sn * offset])
        return jnp.array([chi[0], p[0], p[1]])

    def _forward_platform_from_xi(self, xi: Array, segment_idx: Array) -> Array:
        chi = self._forward_backbone_from_xi(xi, self.L_cum[segment_idx + 1])
        offset = self.distal_cap_length[segment_idx] + self.platform_dimension[segment_idx, 1] / 2.0
        c, sn = jnp.cos(chi[0]), jnp.sin(chi[0])
        p = chi[1:] + jnp.array([-sn * offset, c * offset])
        return jnp.array([chi[0], p[0], p[1]])

    def _forward_end_effector_from_xi(self, xi: Array) -> Array:
        chi_tip = self._forward_backbone_from_xi(xi, self.length)
        last = self.num_segments - 1
        offset = self.distal_cap_length[last] + self.platform_dimension[last, 1]
        c, sn = jnp.cos(chi_tip[0]), jnp.sin(chi_tip[0])
        p_platform_end = chi_tip[1:] + jnp.array([-sn * offset, c * offset])
        p_ee = p_platform_end + jnp.array(
            [
                c * self.end_effector_offset[0] - sn * self.end_effector_offset[1],
                sn * self.end_effector_offset[0] + c * self.end_effector_offset[1],
            ]
        )
        return jnp.array([chi_tip[0] + self.end_effector_offset[2], p_ee[0], p_ee[1]])

    def _kinematic_xi(self, q: Array, eps: Array | None = None) -> Array:
        if eps is None:
            eps = self.global_eps
        return self.apply_eps_to_bend_strains(self.strain(q), eps)

    def forward_kinematics_virtual_backbone(self, q: Array, s: Array) -> Array:
        return self._forward_backbone_from_xi(self._kinematic_xi(q), s)

    _forward_kinematics = forward_kinematics_virtual_backbone

    def forward_kinematics_tips(self, q: Array) -> Array:
        return vmap(lambda s: self.forward_kinematics_virtual_backbone(q, s))(
            self.L_cum[1:]
        )

    def forward_kinematics_batched(self, q: Array, s_ps: Array) -> Array:
        return vmap(lambda s: self.forward_kinematics_virtual_backbone(q, s))(s_ps)

    def _forward_kinematics_arc_length_derivative(self, q: Array, s: Array) -> Array:
        return jax.jvp(
            lambda s_: self._forward_kinematics(q, s_),
            (s,),
            (jnp.ones_like(jnp.asarray(s)),),
        )[1]

    def _forward_kinematics_and_arc_length_derivative(
        self, q: Array, s: Array
    ) -> tuple[Array, Array]:
        return self._forward_kinematics(q, s), self._forward_kinematics_arc_length_derivative(q, s)

    def forward_kinematics_rod(self, q: Array, s: Array, rod_idx: Array) -> Array:
        return self._forward_rod_from_xi(self._kinematic_xi(q), s, rod_idx)

    def forward_kinematics_platform(self, q: Array, segment_idx: Array) -> Array:
        return self._forward_platform_from_xi(self._kinematic_xi(q), segment_idx)

    def forward_kinematics_end_effector(self, q: Array) -> Array:
        return self._forward_end_effector_from_xi(self._kinematic_xi(q))

    def jacobian_virtual_backbone(self, q: Array, s: Array) -> Array:
        xi = self._kinematic_xi(q)
        J_xi = jax.jacfwd(lambda xi_: self._forward_backbone_from_xi(xi_, s))(xi)
        return J_xi @ self.B_xi

    def _jacobian(self, q: Array, s: Array) -> Array:
        return self.jacobian_virtual_backbone(q, s)

    def jacobian_tips(self, q: Array) -> Array:
        return vmap(lambda s: self.jacobian_virtual_backbone(q, s))(self.L_cum[1:])

    def jacobian_batched(self, q: Array, s_ps: Array) -> Array:
        return vmap(lambda s: self.jacobian_virtual_backbone(q, s))(s_ps)

    def _jacobian_arc_length_derivative(self, q: Array, s: Array) -> Array:
        return jax.jvp(
            lambda s_: self._jacobian(q, s_),
            (s,),
            (jnp.ones_like(jnp.asarray(s)),),
        )[1]

    def _jacobian_and_arc_length_derivative(
        self, q: Array, s: Array
    ) -> tuple[Array, Array]:
        return self._jacobian(q, s), self._jacobian_arc_length_derivative(q, s)

    def jacobian_end_effector(self, q: Array) -> Array:
        xi = self._kinematic_xi(q)

        def ee_position_first(xi_: Array) -> Array:
            ee = self._forward_end_effector_from_xi(xi_)
            return jnp.array([ee[1], ee[2], ee[0]])

        return jax.jacfwd(ee_position_first)(xi) @ self.B_xi

    def jacobian_and_time_derivative_virtual_backbone(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        return jax.jvp(
            lambda q_: self.jacobian_virtual_backbone(q_, s), (q,), (qd,)
        )

    def _jacobian_and_time_derivative(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        return self.jacobian_and_time_derivative_virtual_backbone(q, qd, s)

    def jacobian_and_time_derivative_batched(
        self, q: Array, qd: Array, s_ps: Array
    ) -> tuple[Array, Array]:
        return vmap(
            lambda s: self.jacobian_and_time_derivative_virtual_backbone(q, qd, s)
        )(s_ps)

    def inverse_kinematics_end_effector(self, chiee: Array) -> Array:
        if self.num_segments != 1:
            raise AssertionError("Inverse kinematics only works for one segment!")
        hp = self.platform_dimension[0, 1]
        proximal_cap_length = self.proximal_cap_length[0]
        distal_cap_length = self.distal_cap_length[0]
        offset = self.end_effector_offset
        T_b_to_pe = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, proximal_cap_length], [0.0, 0.0, 1.0]])
        T_b_to_ee = jnp.array(
            [
                [jnp.cos(chiee[0]), -jnp.sin(chiee[0]), chiee[1]],
                [jnp.sin(chiee[0]), jnp.cos(chiee[0]), chiee[2]],
                [0.0, 0.0, 1.0],
            ]
        )
        T_de_to_ee = jnp.array(
            [
                [jnp.cos(offset[0]), -jnp.sin(offset[0]), offset[1]],
                [jnp.sin(offset[0]), jnp.cos(offset[0]), distal_cap_length + hp + offset[2]],
                [0.0, 0.0, 1.0],
            ]
        )
        T_pe_to_de = jnp.linalg.inv(T_b_to_pe) @ T_b_to_ee @ jnp.linalg.inv(T_de_to_ee)
        th, px, py = (
            jnp.arctan2(T_pe_to_de[1, 0], T_pe_to_de[0, 0]),
            T_pe_to_de[0, 2],
            T_pe_to_de[1, 2],
        )
        sign = jnp.where(jnp.sign(th) == 0, 1.0, jnp.sign(th))
        th_eps = th + sign * self.global_eps
        xi = th_eps / (2.0 * self.length) * jnp.array(
            [
                2.0,
                py - px * jnp.sin(th_eps) / (jnp.cos(th_eps) - 1.0),
                -px - py * jnp.sin(th_eps) / (jnp.cos(th_eps) - 1.0),
            ]
        )
        return jnp.linalg.pinv(self.B_xi) @ (xi - self.ref_strains())

    def _quadrature(self) -> tuple[Array, Array]:
        points = self.integration_points[1:-1]
        weights = self.integration_weights[1:-1]
        starts = self.L_cum[:-1]
        return (
            starts[:, None] + self.L[:, None] * points[None, :],
            self.L[:, None] * weights[None, :],
        )

    def _rod_strain_mapping(self) -> Array:
        """Return the per-rod map from virtual to physical strains."""
        mapping = jnp.broadcast_to(
            jnp.eye(3, dtype=self.L.dtype),
            (self.num_segments, self.num_rods_per_segment, 3, 3),
        )
        return mapping.at[..., 2, 0].set(self.rod_offset)

    def _platform_mass_properties(self, i: int | Array) -> tuple[Array, Array, Array]:
        width, height, depth = self.platform_dimension[i]
        m_platform = self.platform_density[i] * width * height * depth
        I_platform = m_platform / 12.0 * (width**2 + height**2)
        m_distal = jnp.sum(
            self.end_cap_density[i] * self.distal_cap_length[i] * self.rod_outer_radius[i] ** 2
        )
        I_distal = jnp.sum(
            self.end_cap_density[i]
            * self.distal_cap_length[i]
            * self.rod_outer_radius[i] ** 2
            / 12.0
            * (3.0 * self.rod_outer_radius[i] ** 2 + self.distal_cap_length[i] ** 2)
        )

        if self.num_segments == 1:
            m_proximal = jnp.asarray(0.0, dtype=self.L.dtype)
            I_proximal = jnp.asarray(0.0, dtype=self.L.dtype)
        else:
            next_i = jnp.minimum(i + 1, self.num_segments - 1)
            m_proximal = jnp.where(
                i < self.num_segments - 1,
                jnp.sum(
                    self.end_cap_density[next_i]
                    * self.proximal_cap_length[next_i]
                    * self.rod_outer_radius[next_i] ** 2
                ),
                0.0,
            )
            I_proximal = jnp.where(
                i < self.num_segments - 1,
                jnp.sum(
                    self.end_cap_density[next_i]
                    * self.proximal_cap_length[next_i]
                    * self.rod_outer_radius[next_i] ** 2
                    / 12.0
                    * (3.0 * self.rod_outer_radius[next_i] ** 2 + self.proximal_cap_length[next_i] ** 2)
                ),
                0.0,
            )

        mass = m_platform + m_distal + m_proximal
        rel_y_num = m_distal * self.distal_cap_length[i] / 2.0 + m_platform * (
            self.distal_cap_length[i] + height / 2.0
        )
        rel_y_num = rel_y_num + jnp.where(
            i < self.num_segments - 1,
            m_proximal * (self.distal_cap_length[i] + height + self.proximal_cap_length[i + 1] / 2.0),
            0.0,
        )
        rel_cog = jnp.array([0.0, rel_y_num / mass])
        d_dc = jnp.array([0.0, self.distal_cap_length[i] / 2.0]) - rel_cog
        d_pl = jnp.array([0.0, self.distal_cap_length[i] + height / 2.0]) - rel_cog
        d_pc = jnp.where(
            i < self.num_segments - 1,
            jnp.array([0.0, self.proximal_cap_length[i + 1] / 2.0]) - rel_cog,
            jnp.zeros((2,), dtype=self.L.dtype),
        )
        inertia = (
            I_distal
            + I_platform
            + I_proximal
            + m_distal * (d_dc @ d_dc)
            + m_platform * (d_pl @ d_pl)
            + m_proximal * (d_pc @ d_pc)
        )
        return mass, inertia, rel_cog

    def _rod_mass_matrix_terms(self, xi: Array, i: int, j: int) -> Array:
        area = jnp.pi * (
            self.rod_outer_radius[i, j] ** 2 - self.rod_inner_radius[i, j] ** 2
        )
        moment = jnp.pi / 4.0 * (
            self.rod_outer_radius[i, j] ** 4 - self.rod_inner_radius[i, j] ** 4
        )
        s_points, weights = self._quadrature()

        def rod_pose(z: Array, s: Array) -> Array:
            return self._forward_rod_from_xi(z, s, j)

        jacobian = jax.jacfwd(rod_pose, argnums=0)
        J = vmap(lambda s: jacobian(xi, s))(s_points[i])
        linear = jnp.einsum("kri,krj->kij", J[:, 1:, :], J[:, 1:, :])
        angular = jnp.einsum("ki,kj->kij", J[:, 0, :], J[:, 0, :])
        terms = weights[i, :, None, None] * self.rod_density[i, j] * (
            area * linear + moment * angular
        )
        return jnp.sum(terms, axis=0)

    def _platform_mass_matrix_term(self, xi: Array, i: int) -> Array:
        mass, inertia, rel_cog = self._platform_mass_properties(i)
        s_tip = self.L_cum[i + 1]

        def cog_pose(z: Array) -> Array:
            tip = self._forward_backbone_from_xi(z, s_tip)
            c, sn = jnp.cos(tip[0]), jnp.sin(tip[0])
            p = tip[1:] + jnp.array(
                [c * rel_cog[0] - sn * rel_cog[1], sn * rel_cog[0] + c * rel_cog[1]]
            )
            return jnp.array([tip[0], p[0], p[1]])

        J = jax.jacfwd(cog_pose)(xi)
        return mass * J[1:, :].T @ J[1:, :] + inertia * J[0:1, :].T @ J[0:1, :]

    def _payload_mass_matrix_term(self, xi: Array) -> Array:
        def payload_position(z: Array) -> Array:
            ee = self._forward_end_effector_from_xi(z)
            c, sn = jnp.cos(ee[0]), jnp.sin(ee[0])
            return ee[1:] + jnp.array(
                [
                    c * self.platform_center_of_gravity[0]
                    - sn * self.platform_center_of_gravity[1],
                    sn * self.platform_center_of_gravity[0]
                    + c * self.platform_center_of_gravity[1],
                ]
            )

        J = jax.jacfwd(payload_position)(xi)
        return self.platform_mass * J.T @ J

    def _inertia_full_from_xi(self, xi: Array) -> Array:
        indices = jnp.arange(self.num_segments, dtype=jnp.int32)
        rod_terms = vmap(
            lambda i: vmap(
                lambda j: self._rod_mass_matrix_terms(xi, i, j)
            )(jnp.arange(self.num_rods_per_segment, dtype=jnp.int32))
        )(indices)
        platform_terms = vmap(
            lambda i: self._platform_mass_matrix_term(xi, i)
        )(indices)
        return (
            jnp.sum(rod_terms, axis=(0, 1))
            + jnp.sum(platform_terms, axis=0)
            + self._payload_mass_matrix_term(xi)
        )

    def _dynamic_xi(self, q: Array, eps: Array | None = None) -> Array:
        if eps is None:
            eps = 1e4 * self.global_eps
        return self.apply_eps_to_bend_strains(self.strain(q), eps)

    def _inertia_full_matrix(self, q: Array, eps: float | None = None) -> Array:
        return self._inertia_full_from_xi(self._dynamic_xi(q, eps))

    def inertia_matrix(self, q: Array, eps: float | None = None) -> Array:
        return self.B_xi.T @ self._inertia_full_matrix(q, eps) @ self.B_xi

    def _coriolis_full_matrix(
        self, q: Array, qd: Array, eps: float | None = None
    ) -> Array:
        xi = self._dynamic_xi(q, eps)
        xid = self.B_xi @ qd
        dB = jax.jacfwd(self._inertia_full_from_xi)(xi)
        return 0.5 * jnp.einsum(
            "abk,k->ab",
            dB + jnp.swapaxes(dB, 1, 2) - jnp.transpose(dB, (2, 1, 0)),
            xid,
        )

    def coriolis_matrix(
        self, q: Array, qd: Array, eps: float | None = None
    ) -> Array:
        return self.B_xi.T @ self._coriolis_full_matrix(q, qd, eps) @ self.B_xi

    def _gravitational_energy_full_from_xi(self, xi: Array) -> Array:
        gravity = self.g[1:]
        # The potential is referenced to the base origin and therefore omits
        # the constant energy offset caused by a translated base. Forces are
        # unaffected, but preserving that offset convention keeps energy
        # outputs compatible with the previous model.
        base_translation = self.base_pose[1:]
        s_points, weights = self._quadrature()
        def segment_energy(i: Array) -> Array:
            def rod_energy(j: Array) -> Array:
                area = jnp.pi * (
                    self.rod_outer_radius[i, j] ** 2
                    - self.rod_inner_radius[i, j] ** 2
                )
                rods = vmap(
                    lambda s: self._forward_rod_from_xi(xi, s, j)
                )(s_points[i])
                height = jnp.sum(
                    weights[i]
                    * ((rods[:, 1:] - base_translation) @ gravity)
                )
                return -self.rod_density[i, j] * area * height

            rod_energy_total = jnp.sum(
                vmap(rod_energy)(
                    jnp.arange(self.num_rods_per_segment, dtype=jnp.int32)
                )
            )
            mass, _, rel_cog = self._platform_mass_properties(i)
            tip = self._forward_backbone_from_xi(xi, self.L_cum[i + 1])
            c, sn = jnp.cos(tip[0]), jnp.sin(tip[0])
            p_cog = tip[1:] + jnp.array(
                [c * rel_cog[0] - sn * rel_cog[1], sn * rel_cog[0] + c * rel_cog[1]]
            )
            return rod_energy_total - mass * (gravity @ (p_cog - base_translation))

        energy = jnp.sum(
            vmap(segment_energy)(
                jnp.arange(self.num_segments, dtype=jnp.int32)
            )
        )

        ee = self._forward_end_effector_from_xi(xi)
        c, sn = jnp.cos(ee[0]), jnp.sin(ee[0])
        payload_cog = ee[1:] + jnp.array(
            [
                c * self.platform_center_of_gravity[0]
                - sn * self.platform_center_of_gravity[1],
                sn * self.platform_center_of_gravity[0]
                + c * self.platform_center_of_gravity[1],
            ]
        )
        return energy - self.platform_mass * (gravity @ (payload_cog - base_translation))

    def _gravitational_energy(self, q: Array, eps: float | None = None) -> Array:
        return self._gravitational_energy_full_from_xi(self._dynamic_xi(q, eps))

    def _gravitational_full_force(self, q: Array, eps: float | None = None) -> Array:
        xi = self._dynamic_xi(q, eps)
        return jax.grad(self._gravitational_energy_full_from_xi)(xi)

    def _gravitational_force(self, q: Array, eps: float | None = None) -> Array:
        return self.B_xi.T @ self._gravitational_full_force(q, eps)

    def _nominal_stiffness_full(self) -> Array:
        zeros = jnp.zeros_like(self.nominal_bending_stiffness)
        Shat_rod = jnp.stack(
            [
                jnp.stack(
                    [
                        self.nominal_bending_stiffness,
                        self.bending_shear_stiffness,
                        zeros,
                    ],
                    axis=-1,
                ),
                jnp.stack(
                    [
                        self.bending_shear_stiffness,
                        self.nominal_shear_stiffness,
                        zeros,
                    ],
                    axis=-1,
                ),
                jnp.stack(
                    [zeros, zeros, self.nominal_axial_stiffness], axis=-1
                ),
            ],
            axis=-2,
        )
        mapping = self._rod_strain_mapping()
        blocks = jnp.einsum("...ji,...jk,...kl->...il", mapping, Shat_rod, mapping)
        return blk_diag(jnp.sum(blocks, axis=1))

    def Shat(self) -> Array:
        return self._nominal_stiffness_full()

    def stiffness_matrix(self) -> Array:
        return self.B_xi.T @ self.Shat() @ self.B_xi

    def _stiffness_full_vector(self, q: Array) -> Array:
        return self.Shat() @ (self.strain(q) - self.ref_strains())

    def elastic_force(self, q: Array) -> Array:
        return self.B_xi.T @ self._stiffness_full_vector(q)

    def _elastic_energy(self, q: Array) -> Array:
        return 0.5 * q @ self.stiffness_matrix() @ q

    def _damping_full_matrix(self) -> Array:
        zeros = jnp.zeros_like(self.bending_damping)
        damping_rod = jnp.stack(
            [
                jnp.stack(
                    [self.bending_damping, zeros, zeros], axis=-1
                ),
                jnp.stack(
                    [zeros, self.shear_damping, zeros], axis=-1
                ),
                jnp.stack(
                    [zeros, zeros, self.axial_damping], axis=-1
                ),
            ],
            axis=-2,
        )
        mapping = self._rod_strain_mapping()
        blocks = jnp.einsum(
            "...ji,...jk,...kl->...il", mapping, damping_rod, mapping
        )
        return blk_diag(jnp.sum(blocks, axis=1))

    def damping_matrix(self, q: Array) -> Array:
        del q
        return self.B_xi.T @ self._damping_full_matrix() @ self.B_xi

    def _actuation_full_matrix(self, q: Array, phi: Array) -> Array:
        xi_rows = self.strain(q).reshape(self.num_segments, 3)
        physical = self.beta(xi_rows.reshape(-1))
        references = self._reference_physical_strains()
        phi = jnp.asarray(phi).reshape(-1)
        scale = (
            self.rod_height / self.L[:, None]
        ) * phi.reshape(self.num_segments, self.num_rods_per_segment)
        delta = jnp.stack(
            [
                self.bending_stiffness_correction * scale,
                self.shear_stiffness_correction * scale,
                self.axial_stiffness_correction * scale,
            ],
            axis=-1,
        )
        zeros = jnp.zeros_like(delta[..., 0])
        Sr = jnp.stack(
            [
                jnp.stack(
                    [
                        self.nominal_bending_stiffness + delta[..., 0],
                        self.bending_shear_stiffness,
                        zeros,
                    ],
                    axis=-1,
                ),
                jnp.stack(
                    [
                        self.bending_shear_stiffness,
                        self.nominal_shear_stiffness + delta[..., 1],
                        zeros,
                    ],
                    axis=-1,
                ),
                jnp.stack(
                    [zeros, zeros, self.nominal_axial_stiffness + delta[..., 2]],
                    axis=-1,
                ),
            ],
            axis=-2,
        )
        coupling = jnp.stack(
            [zeros, zeros, self.strain_coupling * scale], axis=-1
        )
        physical_error = physical - references
        rod_force_physical = -delta * physical_error + jnp.einsum(
            "...jk,...k->...j", Sr, coupling
        )
        rod_force_virtual = jnp.einsum(
            "...ji,...j->...i", self._rod_strain_mapping(), rod_force_physical
        )
        return jnp.sum(rod_force_virtual, axis=1).reshape(self.num_strains)

    def actuation_force(
        self, q: Array, phi: Array, qd: Array | None = None
    ) -> Array:
        del qd
        if not self.consider_underactuation:
            return jnp.asarray(phi)
        return self.B_xi.T @ self._actuation_full_matrix(q, phi)

    def hysteresis_force(self, q: Array, z: Array) -> Array:
        del q
        return self.B_xi.T @ self.Shat() @ (self.hysteresis_basis @ z)

    def dynamics_terms(self, q: Array, qd: Array) -> tuple[Array, Array, Array]:
        B = self.inertia_matrix(q)
        Cqd = self.coriolis_matrix(q, qd) @ qd
        G = self._gravitational_force(q)
        return B, Cqd, G

    def forward_dynamics(
        self, t: Array, y: Array, actuation_args: tuple[Array, Array | None]
    ) -> Array:
        """Compute the autonomous HSA state derivative.

        ``t`` is part of the Diffrax/dynamical-system callback signature, but
        this model has no explicit time-dependent parameters or inputs.
        """
        del t
        u, tau_ext = actuation_args
        if tau_ext is None:
            tau_ext = jnp.zeros((self.num_dofs,), dtype=y.dtype)
        if self.consider_hysteresis:
            q, qd, z = jnp.split(y, [self.num_dofs, 2 * self.num_dofs])
            hysteresis_basis_active = self.B_xi.T @ self.hysteresis_basis
            zd = (hysteresis_basis_active.T @ qd) * (
                self.hysteresis_A
                - jnp.abs(z) ** self.hysteresis_n
                * (
                    self.hysteresis_gamma
                    + self.hysteresis_beta
                    * jnp.sign((hysteresis_basis_active.T @ qd) * z)
                )
            )
        else:
            q, qd = jnp.split(y, [self.num_dofs])
            z = jnp.zeros((0,), dtype=y.dtype)
            zd = z

        tau_u = self.actuation_force(q, u) if self.consider_underactuation else jnp.asarray(u)
        if self.consider_hysteresis:
            tau_el_full = self._stiffness_full_vector(q)
            tau_hyst_full = self.Shat() @ (self.hysteresis_basis @ z)
            tau_el = self.B_xi.T @ (
                self.hysteresis_alpha * tau_el_full
                + (1.0 - self.hysteresis_alpha) * tau_hyst_full
            )
        else:
            tau_el = self.elastic_force(q)
        B, Cqd, G = self.dynamics_terms(q, qd)
        qdd = jnp.linalg.solve(
            B, tau_u + tau_ext - Cqd - G - tau_el - self.damping_matrix(q) @ qd
        )
        return (
            jnp.concatenate((qd, qdd, zd))
            if self.consider_hysteresis
            else jnp.concatenate((qd, qdd))
        )
