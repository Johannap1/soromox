__all__ = ["PlanarHSA"]


from typing import Any

import equinox as eqx
from jax import Array, lax, vmap
from jax import numpy as jnp

from soromox.systems.pcs.params import PlanarHSAParams
from soromox.systems.pcs.planar_pcs import PlanarPCS
from soromox.systems.pcs.structures import PlanarHSAStructure
from soromox.systems.soft_robot import CrossSectionGeometry, SoftRobot
from soromox.utils.array_math import blk_diag
from soromox.utils.dof import build_active_dof_basis
from soromox.utils.geometry import poses
from soromox.utils.integration import (
    gauss_quadrature,
    scale_interior_gaussian_quadrature,
)
from soromox.utils.lie_algebra import constant_strain, se2


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
    are evaluated numerically with JAX and support JIT compilation.
    Optional underactuation maps motor inputs to rod forces, while optional
    Bouc--Wen hysteresis augments the elastic response.

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
        xi_ref: Reference virtual-backbone strain.
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

    # Configuration-independent HSA quantities.  Keeping these as model
    # leaves follows the cache pattern used by PCS and avoids rebuilding the
    # same small arrays in every jitted dynamics call.
    rod_strain_mapping: Array
    rod_transforms: Array
    rod_adjoint_inverses: Array
    rod_mass_matrices: Array
    segment_masses: Array
    segment_inertias: Array
    segment_rel_cogs: Array
    cog_transforms: Array
    cog_adjoint_inverses: Array
    payload_transform: Array
    payload_adjoint_inverse: Array
    payload_mass_matrix: Array

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
        self.xi_ref = self.beta_inv(self._reference_physical_strains())
        self._set_hysteresis(params)
        self._refresh_hsa_caches()
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

    def _refresh_hsa_caches(self) -> None:
        """Refresh HSA quantities that do not depend on the configuration."""
        mapping = self._build_rod_strain_mapping()
        rod_transforms = vmap(
            lambda offsets: vmap(
                lambda offset: self._translation_transform(
                    offset, jnp.zeros_like(offset)
                )
            )(offsets)
        )(self.rod_offset)
        rod_adjoint_inverses = vmap(vmap(se2.adjoint_inverse))(rod_transforms)

        area = jnp.pi * (
            self.rod_outer_radius**2 - self.rod_inner_radius**2
        )
        second_moment = jnp.pi / 4.0 * (
            self.rod_outer_radius**4 - self.rod_inner_radius**4
        )
        rod_mass_matrices = jnp.zeros((*area.shape, 3, 3), dtype=area.dtype)
        rod_mass_matrices = rod_mass_matrices.at[..., 0, 0].set(
            self.rod_density * second_moment
        )
        rod_mass_matrices = rod_mass_matrices.at[..., 1, 1].set(
            self.rod_density * area
        )
        rod_mass_matrices = rod_mass_matrices.at[..., 2, 2].set(
            self.rod_density * area
        )

        segment_masses, segment_inertias, segment_rel_cogs = vmap(
            self._platform_mass_properties
        )(jnp.arange(self.num_segments, dtype=jnp.int32))
        cog_transforms = vmap(
            lambda rel_cog: self._translation_transform(
                jnp.zeros_like(rel_cog[1]), rel_cog[1]
            )
        )(segment_rel_cogs)
        cog_adjoint_inverses = vmap(se2.adjoint_inverse)(cog_transforms)

        last = self.num_segments - 1
        platform_end = self._translation_transform(
            jnp.zeros_like(self.distal_cap_length[last]),
            self.distal_cap_length[last] + self.platform_dimension[last, 1],
        )
        payload_cog = self._translation_transform(
            self.platform_center_of_gravity[0],
            self.platform_center_of_gravity[1],
        )
        payload_transform = (
            platform_end
            @ poses.planar_pose_to_transform(self.end_effector_offset)
            @ payload_cog
        )
        payload_adjoint_inverse = se2.adjoint_inverse(payload_transform)
        payload_mass_matrix = jnp.diag(
            jnp.array([0.0, self.platform_mass, self.platform_mass], dtype=self.L.dtype)
        )

        K_full = self._build_nominal_stiffness_full(mapping)
        K_active = self.B_xi.T @ K_full @ self.B_xi
        D_full = self._build_damping_full(mapping)
        D_active = self.B_xi.T @ D_full @ self.B_xi

        for name, value in (
            ("rod_strain_mapping", mapping),
            ("rod_transforms", rod_transforms),
            ("rod_adjoint_inverses", rod_adjoint_inverses),
            ("rod_mass_matrices", rod_mass_matrices),
            ("segment_masses", segment_masses),
            ("segment_inertias", segment_inertias),
            ("segment_rel_cogs", segment_rel_cogs),
            ("cog_transforms", cog_transforms),
            ("cog_adjoint_inverses", cog_adjoint_inverses),
            ("payload_transform", payload_transform),
            ("payload_adjoint_inverse", payload_adjoint_inverse),
            ("payload_mass_matrix", payload_mass_matrix),
            ("K_full", K_full),
            ("K_active", K_active),
            ("D_full", D_full),
            ("D_active", D_active),
        ):
            object.__setattr__(self, name, value)

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
            lambda model: model.xi_ref,
            updated,
            updated.beta_inv(updated._reference_physical_strains()),
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
        updated._refresh_hsa_caches()
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
        return jnp.asarray(self.params.xi_ref, dtype=self.L.dtype)

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
        return self.xi_ref

    def strain(self, q: Array) -> Array:
        return self.B_xi @ q + self.xi_ref

    @staticmethod
    def _translation_transform(x: Array, y: Array) -> Array:
        return poses.planar_pose_to_transform(
            jnp.stack([jnp.zeros_like(x), x, y])
        )

    def _segment_bases_and_tips(self, xi: Array) -> tuple[Array, Array]:
        xi = xi.reshape(self.num_segments, 3)
        g0 = poses.planar_pose_to_transform(jnp.asarray(self.base_pose, dtype=xi.dtype))

        def scan_body(g_prev: Array, i: Array) -> tuple[Array, tuple[Array, Array]]:
            # The established HSA geometry places proximal-cap lengths along
            # the world y-axis. Keep that convention while reusing the PCS
            # exponential for the flexible section and the local post-section
            # transform below.
            g_base = g_prev.at[1, 2].add(self.proximal_cap_length[i])
            xi_i = xi[i]
            g_tip = g_base @ se2.exp(self.L[i] * xi_i, eps=self.global_eps)
            post = self._translation_transform(
                jnp.zeros_like(self.distal_cap_length[i]),
                self.distal_cap_length[i] + self.platform_dimension[i, 1],
            )
            return g_tip @ post, (g_base, g_tip)

        _, (g_bases, g_tips) = lax.scan(
            scan_body,
            g0,
            jnp.arange(self.num_segments, dtype=jnp.int32),
        )
        return g_bases, g_tips

    def _backbone_transform(self, xi: Array, s: Array) -> Array:
        segment_idx, s_local = self.classify_segment(s)
        g_bases, _ = self._segment_bases_and_tips(xi)
        g_base = g_bases[segment_idx]
        return g_base @ se2.exp(s_local * xi.reshape(self.num_segments, 3)[segment_idx], eps=self.global_eps)

    def _forward_backbone_from_xi(self, xi: Array, s: Array) -> Array:
        return poses.planar_pose_from_transform(
            self._backbone_transform(xi, s), eps=self.global_eps
        )

    def _forward_rod_from_xi(self, xi: Array, s: Array, rod_idx: Array) -> Array:
        segment_idx, _ = self.classify_segment(s)
        offset = self.rod_offset[segment_idx, rod_idx]
        rod_transform = self._translation_transform(offset, jnp.zeros_like(offset))
        return poses.planar_pose_from_transform(
            self._backbone_transform(xi, s) @ rod_transform,
            eps=self.global_eps,
        )

    def _forward_platform_from_xi(self, xi: Array, segment_idx: Array) -> Array:
        _, g_tips = self._segment_bases_and_tips(xi)
        offset = self.distal_cap_length[segment_idx] + self.platform_dimension[segment_idx, 1] / 2.0
        platform_transform = self._translation_transform(
            jnp.zeros_like(offset), offset
        )
        return poses.planar_pose_from_transform(
            g_tips[segment_idx] @ platform_transform, eps=self.global_eps
        )

    def _end_effector_transform_from_tip(self, g_tip: Array) -> Array:
        last = self.num_segments - 1
        platform_end = self._translation_transform(
            jnp.zeros_like(self.distal_cap_length[last]),
            self.distal_cap_length[last] + self.platform_dimension[last, 1],
        )
        return g_tip @ platform_end @ poses.planar_pose_to_transform(self.end_effector_offset)

    def _forward_end_effector_from_xi(self, xi: Array) -> Array:
        _, g_tips = self._segment_bases_and_tips(xi)
        return poses.planar_pose_from_transform(
            self._end_effector_transform_from_tip(g_tips[-1]), eps=self.global_eps
        )

    def _kinematic_xi(self, q: Array) -> Array:
        return self.strain(q)

    def _forward_kinematics(self, q: Array, s: Array) -> Array:
        return self._forward_backbone_from_xi(self._kinematic_xi(q), s)

    def _forward_kinematics_arc_length_derivative(self, q: Array, s: Array) -> Array:
        xi = self._kinematic_xi(q).reshape(self.num_segments, 3)
        segment_idx, _ = self.classify_segment(s)
        g = self._backbone_transform(self._kinematic_xi(q), s)
        return jnp.concatenate(
            [
                xi[segment_idx, :1],
                g[:2, :2] @ xi[segment_idx, 1:],
            ]
        )

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

    def _segment_bases_and_jacobians_geometry(
        self, xi: Array
    ) -> tuple[Array, Array, Array, Array]:
        """Propagate poses and Jacobians without time-derivative terms."""
        xi = xi.reshape(self.num_segments, 3)
        zeros = jnp.zeros((self.num_segments, 3, 3), dtype=xi.dtype)
        g0 = poses.planar_pose_to_transform(jnp.asarray(self.base_pose, dtype=xi.dtype))

        def scan_body(
            carry: tuple[Array, Array], i: Array
        ) -> tuple[tuple[Array, Array], tuple[Array, Array, Array, Array]]:
            g_prev, J_prev = carry
            g_base = g_prev.at[1, 2].add(self.proximal_cap_length[i])
            J_base = J_prev
            xi_i = xi[i]
            g_tip = g_base @ se2.exp(self.L[i] * xi_i, eps=self.global_eps)
            Ad_inv, T = self._pcs_jacobian_step_terms(xi_i, self.L[i])
            J_tip = self._update_body_jacobian_step(J_base, i, Ad_inv, T)
            post = self._translation_transform(
                jnp.zeros_like(self.distal_cap_length[i]),
                self.distal_cap_length[i] + self.platform_dimension[i, 1],
            )
            Ad_inv_post = se2.adjoint_inverse(post)
            return (
                g_tip @ post,
                Ad_inv_post @ J_tip,
            ), (g_base, J_base, g_tip, J_tip)

        _, values = lax.scan(
            scan_body,
            (g0, zeros),
            jnp.arange(self.num_segments, dtype=jnp.int32),
        )
        return values

    def _integration_geometry_full(
        self, q: Array
    ) -> tuple[Array, ...]:
        """Compute configuration-dependent integration geometry once."""
        xi = self.strain(q).reshape(self.num_segments, 3)
        Xs_scaled, Ws_scaled = vmap(
            scale_interior_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.integration_points,
            self.integration_weights,
            self.L_cum[:-1],
            self.L_cum[1:],
        )
        s_local = Xs_scaled - self.L_cum[:-1, None]
        g_bases, J_bases, g_tips, J_tips = (
            self._segment_bases_and_jacobians_geometry(xi)
        )

        def segment_kinematics(
            i: Array,
            xi_i: Array,
            g_base: Array,
            J_base: Array,
            s_local_i: Array,
        ) -> tuple[Array, Array]:
            def point_kinematics(s_local_ij: Array) -> tuple[Array, Array]:
                Ad_inv, T = self._pcs_jacobian_step_terms(xi_i, s_local_ij)
                g_ij = g_base @ se2.exp(s_local_ij * xi_i, eps=self.global_eps)
                J_ij = self._update_body_jacobian_step(
                    J_base, i, Ad_inv, T
                )
                return g_ij, self._final_size_jacobian(J_ij)

            return vmap(point_kinematics)(s_local_i)

        g_ps, J_ps = vmap(segment_kinematics)(
            jnp.arange(self.num_segments, dtype=jnp.int32),
            xi,
            g_bases,
            J_bases,
            s_local,
        )
        return (
            Ws_scaled,
            g_ps,
            J_ps,
            g_bases,
            J_bases,
            xi,
            s_local,
            g_tips,
            J_tips,
        )

    def _segment_jacobian_time_derivatives(
        self,
        xi: Array,
        xid: Array,
        J_bases: Array,
    ) -> tuple[Array, Array]:
        """Propagate segment-tip Jacobian time derivatives for one velocity."""
        xi = xi.reshape(self.num_segments, 3)
        xid = xid.reshape(self.num_segments, 3)
        zeros = jnp.zeros((self.num_segments, 3, 3), dtype=xi.dtype)

        def scan_body(
            Jd_prev: Array, i: Array
        ) -> tuple[Array, tuple[Array, Array]]:
            Jd_base = Jd_prev
            xi_i = xi[i]
            xid_i = xid[i]
            Ad_inv, T = self._pcs_jacobian_step_terms(xi_i, self.L[i])
            Td = constant_strain.tangent_derivative_se2(
                xi_i, xid_i, self.L[i], eps=self.tangent_eps
            )
            Jd_tip = self._update_body_jacobian_time_derivative_step(
                J_bases[i], Jd_base, i, xid_i, Ad_inv, T, Td
            )[1]
            post = self._translation_transform(
                jnp.zeros_like(self.distal_cap_length[i]),
                self.distal_cap_length[i] + self.platform_dimension[i, 1],
            )
            return se2.adjoint_inverse(post) @ Jd_tip, (Jd_base, Jd_tip)

        _, values = lax.scan(
            scan_body,
            zeros,
            jnp.arange(self.num_segments, dtype=jnp.int32),
        )
        return values

    def _integration_jacobian_time_derivative_from_geometry(
        self,
        qd: Array,
        geometry: tuple[Array, ...],
        convective_only_jd: bool = False,
    ) -> tuple[Array, Array]:
        """Compute ``Jdot`` while reusing geometry already built for ``q``."""
        _, _, _, _, J_bases, xi, s_local, _, _ = geometry
        xid = (self.B_xi @ qd).reshape(self.num_segments, 3)
        Jd_bases, Jd_tips = self._segment_jacobian_time_derivatives(
            xi, xid, J_bases
        )

        def segment_derivative(
            i: Array,
            xi_i: Array,
            xid_i: Array,
            J_base: Array,
            Jd_base: Array,
            s_local_i: Array,
        ) -> Array:
            def point_derivative(s_local_ij: Array) -> Array:
                Ad_inv, T = self._pcs_jacobian_step_terms(xi_i, s_local_ij)
                Td = constant_strain.tangent_derivative_se2(
                    xi_i, xid_i, s_local_ij, eps=self.tangent_eps
                )
                Jd_ij = self._update_body_jacobian_time_derivative_step(
                    J_base,
                    Jd_base,
                    i,
                    xid_i,
                    Ad_inv,
                    T,
                    Td,
                )[1]
                if convective_only_jd:
                    eta = (Ad_inv @ T) @ xid_i
                    Ad_inv_dot = -se2.small_adjoint(eta) @ Ad_inv
                    Jd_ij = (
                        jnp.einsum("ab,nbc->nac", Ad_inv, Jd_base)
                        + jnp.einsum("ab,nbc->nac", Ad_inv_dot, J_base)
                    ).at[i].set(Ad_inv @ Td)
                return self._final_size_jacobian(Jd_ij)

            return vmap(point_derivative)(s_local_i)

        Jd_ps = vmap(segment_derivative)(
            jnp.arange(self.num_segments, dtype=jnp.int32),
            xi,
            xid,
            J_bases,
            Jd_bases,
            s_local,
        )
        return Jd_ps, Jd_tips

    def _integration_kinematics_gravity(
        self, q: Array
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Return only pose/Jacobian data needed by gravitational forces."""
        Ws_scaled, g_ps, J_ps, _, _, _, _, g_tips, J_tips = (
            self._integration_geometry_full(q)
        )
        return Ws_scaled, g_ps, J_ps, g_tips, J_tips

    def _mass_geometry_from_integration(
        self, geometry: tuple[Array, ...], full: bool
    ) -> tuple[Array, ...]:
        """Assemble mass-property geometry from a shared integration pass."""
        Ws, g_ps, J_full, _, _, _, _, g_tips, J_tips = geometry
        basis = (
            jnp.eye(self.num_strains, dtype=J_full.dtype)
            if full
            else self.B_xi
        )
        J_ps = jnp.einsum("sqab,bd->sqad", J_full, basis)
        J_rods = jnp.einsum(
            "srab,sqbd->sqrad", self.rod_adjoint_inverses, J_ps
        )
        g_rods = jnp.einsum(
            "sqab,srbc->sqrac", g_ps, self.rod_transforms
        )

        tip_indices = jnp.arange(self.num_segments, dtype=jnp.int32)
        J_tips = J_tips[tip_indices, tip_indices]
        J_tips = jnp.einsum("sab,bd->sad", J_tips, basis)
        J_cogs = jnp.einsum(
            "sab,sbd->sad", self.cog_adjoint_inverses, J_tips
        )
        g_cogs = jnp.einsum("sab,sbc->sac", g_tips, self.cog_transforms)
        J_payload = self.payload_adjoint_inverse @ J_tips[-1]
        g_payload = g_tips[-1] @ self.payload_transform
        return (
            Ws,
            g_rods,
            J_rods,
            self.rod_mass_matrices,
            g_cogs,
            J_cogs,
            self.segment_masses,
            self.segment_inertias,
            J_payload,
            g_payload,
            self.payload_mass_matrix,
        )

    def _segment_bases_and_jacobians(
        self, xi: Array, xid: Array
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        xi = xi.reshape(self.num_segments, 3)
        xid = xid.reshape(self.num_segments, 3)
        zeros = jnp.zeros((self.num_segments, 3, 3), dtype=xi.dtype)
        g0 = poses.planar_pose_to_transform(jnp.asarray(self.base_pose, dtype=xi.dtype))

        def scan_body(
            carry: tuple[Array, Array, Array], i: Array
        ) -> tuple[tuple[Array, Array, Array], tuple[Array, Array, Array, Array, Array, Array]]:
            g_prev, J_prev, Jd_prev = carry
            g_base = g_prev.at[1, 2].add(self.proximal_cap_length[i])
            J_base = J_prev
            Jd_base = Jd_prev

            xi_i = xi[i]
            xid_i = xid[i]
            g_tip = g_base @ se2.exp(self.L[i] * xi_i, eps=self.global_eps)
            Ad_inv, T = self._pcs_jacobian_step_terms(xi_i, self.L[i])
            Td = constant_strain.tangent_derivative_se2(
                xi_i, xid_i, self.L[i], eps=self.tangent_eps
            )
            J_tip, Jd_tip = self._update_body_jacobian_time_derivative_step(
                J_base, Jd_base, i, xid_i, Ad_inv, T, Td
            )

            post = self._translation_transform(
                jnp.zeros_like(self.distal_cap_length[i]),
                self.distal_cap_length[i] + self.platform_dimension[i, 1],
            )
            Ad_inv_post = se2.adjoint_inverse(post)
            return (
                g_tip @ post,
                Ad_inv_post @ J_tip,
                Ad_inv_post @ Jd_tip,
            ), (g_base, J_base, Jd_base, g_tip, J_tip, Jd_tip)

        _, values = lax.scan(
            scan_body,
            (g0, zeros, zeros),
            jnp.arange(self.num_segments, dtype=jnp.int32),
        )
        return values

    def _body_kinematics_at(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array, Array, Array]:
        xi = self.strain(q)
        xid = self.B_xi @ qd
        segment_idx, s_local = self.classify_segment(s)
        g_bases, J_bases, Jd_bases, _, _, _ = self._segment_bases_and_jacobians(xi, xid)
        xi_rows = xi.reshape(self.num_segments, 3)
        xid_rows = xid.reshape(self.num_segments, 3)
        xi_i = xi_rows[segment_idx]
        xid_i = xid_rows[segment_idx]
        J_base = J_bases[segment_idx]
        Jd_base = Jd_bases[segment_idx]
        g_base = g_bases[segment_idx]
        Ad_inv, T = self._pcs_jacobian_step_terms(xi_i, s_local)
        Td = constant_strain.tangent_derivative_se2(
            xi_i, xid_i, s_local, eps=self.tangent_eps
        )
        g = g_base @ se2.exp(s_local * xi_i, eps=self.global_eps)
        J_full, Jd_full = self._update_body_jacobian_time_derivative_step(
            J_base, Jd_base, segment_idx, xid_i, Ad_inv, T, Td
        )
        return (
            g,
            self._final_size_jacobian(J_full) @ self.B_xi,
            self._final_size_jacobian(Jd_full) @ self.B_xi,
            xi_i,
        )

    def _jacobian_bodyframe_with_pose(self, q: Array, s: Array) -> tuple[Array, Array]:
        g, J, _, _ = self._body_kinematics_at(q, jnp.zeros_like(q), s)
        return poses.planar_pose_from_transform(g, eps=self.global_eps), J

    def _jacobian(self, q: Array, s: Array) -> Array:
        chi, J_local = self._jacobian_bodyframe_with_pose(q, s)
        return self._body_jacobian_to_inertial(chi, J_local)

    def _jacobian_arc_length_derivative(self, q: Array, s: Array) -> Array:
        xi = self.strain(q)
        segment_idx, s_local = self.classify_segment(s)
        xi_rows = xi.reshape(self.num_segments, 3)
        xi_i = xi_rows[segment_idx]
        g_bases, J_bases, _, _, _, _ = self._segment_bases_and_jacobians(
            xi, jnp.zeros_like(xi)
        )
        g_base = g_bases[segment_idx]
        J_base = J_bases[segment_idx]
        Ad_inv, T, dAd_inv_ds, dT_ds = self._pcs_jacobian_arc_length_step_terms(
            xi_i, s_local
        )
        g = g_base @ se2.exp(s_local * xi_i, eps=self.global_eps)
        J_full = self._update_body_jacobian_step(J_base, segment_idx, Ad_inv, T)
        J_local = self._final_size_jacobian(J_full) @ self.B_xi
        Js_full = self._update_body_jacobian_arc_length_derivative_step(
            J_base, segment_idx, Ad_inv, T, dAd_inv_ds, dT_ds
        )
        Js_local = self._final_size_jacobian(Js_full) @ self.B_xi
        chi = poses.planar_pose_from_transform(g, eps=self.global_eps)
        Ad_rot = self._rotation_adjoint_from_pose(chi)
        Ad_rot_s = Ad_rot @ se2.small_adjoint(
            jnp.array([xi_i[0], 0.0, 0.0], dtype=xi_i.dtype)
        )
        return Ad_rot_s @ J_local + Ad_rot @ Js_local

    def _jacobian_and_arc_length_derivative(
        self, q: Array, s: Array
    ) -> tuple[Array, Array]:
        return self._jacobian(q, s), self._jacobian_arc_length_derivative(q, s)

    def jacobian_end_effector(self, q: Array) -> Array:
        g_tip, J_tip_local, _, _ = self._body_kinematics_at(
            q, jnp.zeros_like(q), self.length
        )
        last = self.num_segments - 1
        h = self._translation_transform(
            jnp.zeros_like(self.distal_cap_length[last]),
            self.distal_cap_length[last] + self.platform_dimension[last, 1],
        ) @ poses.planar_pose_to_transform(self.end_effector_offset)
        g_ee = g_tip @ h
        J_ee_local = se2.adjoint_inverse(h) @ J_tip_local
        chi_ee = poses.planar_pose_from_transform(g_ee, eps=self.global_eps)
        J_pose = self._body_jacobian_to_inertial(chi_ee, J_ee_local)
        return J_pose[jnp.array([1, 2, 0])]

    def jacobian_and_time_derivative_end_effector(
        self, q: Array, qd: Array
    ) -> tuple[Array, Array]:
        g_tip, J_tip_local, Jd_tip_local, _ = self._body_kinematics_at(
            q, qd, self.length
        )
        last = self.num_segments - 1
        h = self._translation_transform(
            jnp.zeros_like(self.distal_cap_length[last]),
            self.distal_cap_length[last] + self.platform_dimension[last, 1],
        ) @ poses.planar_pose_to_transform(self.end_effector_offset)
        g_ee = g_tip @ h
        Ad_h_inv = se2.adjoint_inverse(h)
        J_ee_local = Ad_h_inv @ J_tip_local
        Jd_ee_local = Ad_h_inv @ Jd_tip_local
        chi_ee = poses.planar_pose_from_transform(g_ee, eps=self.global_eps)
        Ad_rot = self._rotation_adjoint_from_pose(chi_ee)
        eta = J_ee_local @ qd
        Ad_rot_dot = Ad_rot @ se2.small_adjoint(
            jnp.array([eta[0], 0.0, 0.0], dtype=eta.dtype)
        )
        J_pose = Ad_rot @ J_ee_local
        Jd_pose = Ad_rot @ Jd_ee_local + Ad_rot_dot @ J_ee_local
        return (
            J_pose[jnp.array([1, 2, 0])],
            Jd_pose[jnp.array([1, 2, 0])],
        )

    def _jacobian_and_time_derivative(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        g, J_local, Jd_local, _ = self._body_kinematics_at(q, qd, s)
        chi = poses.planar_pose_from_transform(g, eps=self.global_eps)
        Ad_rot = self._rotation_adjoint_from_pose(chi)
        eta = J_local @ qd
        eta_rot = jnp.array([eta[0], 0.0, 0.0], dtype=eta.dtype)
        Ad_rot_dot = Ad_rot @ se2.small_adjoint(eta_rot)
        return (
            Ad_rot @ J_local,
            Ad_rot @ Jd_local + Ad_rot_dot @ J_local,
        )

    def jacobian_and_time_derivative_batched(
        self, q: Array, qd: Array, s_ps: Array
    ) -> tuple[Array, Array]:
        return vmap(lambda s: self.jacobian_and_time_derivative(q, qd, s))(s_ps)

    def inverse_kinematics(self, chi_ee: Array) -> Array:
        if self.num_segments != 1:
            raise AssertionError("Inverse kinematics only works for one segment!")
        # Keep the historical one-segment target inversion, including its
        # small-angle regularization.  The forward and differential paths use
        # the protected Lie-group operators directly; this closed form is only
        # used to preserve the documented inverse-kinematics result.
        hp = self.platform_dimension[0, 1]
        proximal_cap_length = self.proximal_cap_length[0]
        distal_cap_length = self.distal_cap_length[0]
        offset = self.end_effector_offset
        T_b_to_pe = self._translation_transform(
            jnp.zeros_like(proximal_cap_length), proximal_cap_length
        )
        T_b_to_ee = poses.planar_pose_to_transform(chi_ee)
        T_de_to_ee = poses.planar_pose_to_transform(
            jnp.array([offset[0], offset[1], distal_cap_length + hp + offset[2]])
        )
        T_pe_to_de = jnp.linalg.inv(T_b_to_pe) @ T_b_to_ee @ jnp.linalg.inv(T_de_to_ee)
        th = jnp.arctan2(T_pe_to_de[1, 0], T_pe_to_de[0, 0])
        px, py = T_pe_to_de[0, 2], T_pe_to_de[1, 2]
        sign = jnp.where(jnp.sign(th) == 0, 1.0, jnp.sign(th))
        th_eps = th + sign * self.global_eps
        xi = th_eps / (2.0 * self.length) * jnp.array(
            [
                2.0,
                py - px * jnp.sin(th_eps) / (jnp.cos(th_eps) - 1.0),
                -px - py * jnp.sin(th_eps) / (jnp.cos(th_eps) - 1.0),
            ]
        )
        return jnp.linalg.pinv(self.B_xi) @ (xi - self.xi_ref)

    def _quadrature(self) -> tuple[Array, Array]:
        points = self.integration_points[1:-1]
        weights = self.integration_weights[1:-1]
        starts = self.L_cum[:-1]
        return (
            starts[:, None] + self.L[:, None] * points[None, :],
            self.L[:, None] * weights[None, :],
        )

    def _build_rod_strain_mapping(self) -> Array:
        """Build the per-rod map from virtual to physical strains."""
        mapping = jnp.broadcast_to(
            jnp.eye(3, dtype=self.L.dtype),
            (self.num_segments, self.num_rods_per_segment, 3, 3),
        )
        return mapping.at[..., 2, 0].set(self.rod_offset)

    def _rod_strain_mapping(self) -> Array:
        """Return the cached per-rod map from virtual to physical strains."""
        return self.rod_strain_mapping

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

    def _integration_kinematics_full(
        self,
        q: Array,
        qd: Array,
        convective_only_jd: bool = False,
        return_tips: bool = False,
    ) -> tuple[Array, ...]:
        xi = self.strain(q).reshape(self.num_segments, 3)
        xid = (self.B_xi @ qd).reshape(self.num_segments, 3)
        Xs_scaled, Ws_scaled = vmap(
            scale_interior_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.integration_points,
            self.integration_weights,
            self.L_cum[:-1],
            self.L_cum[1:],
        )
        s_local = Xs_scaled - self.L_cum[:-1, None]
        g_bases, J_bases, Jd_bases, g_tips, J_tips, Jd_tips = self._segment_bases_and_jacobians(
            xi, xid
        )

        def segment_kinematics(
            i: Array,
            xi_i: Array,
            xid_i: Array,
            g_base: Array,
            J_base: Array,
            Jd_base: Array,
            s_local_i: Array,
        ) -> tuple[Array, Array, Array]:
            def point_kinematics(s_local_ij: Array) -> tuple[Array, Array, Array]:
                Ad_inv, T = self._pcs_jacobian_step_terms(xi_i, s_local_ij)
                Td = constant_strain.tangent_derivative_se2(
                    xi_i, xid_i, s_local_ij, eps=self.tangent_eps
                )
                g_ij = g_base @ se2.exp(s_local_ij * xi_i, eps=self.global_eps)
                J_ij, Jd_ij = self._update_body_jacobian_time_derivative_step(
                    J_base,
                    Jd_base,
                    i,
                    xid_i,
                    Ad_inv,
                    T,
                    Td,
                )
                if convective_only_jd:
                    eta = (Ad_inv @ T) @ xid_i
                    Ad_inv_dot = -se2.small_adjoint(eta) @ Ad_inv
                    Jd_ij = (
                        jnp.einsum("ab,nbc->nac", Ad_inv, Jd_base)
                        + jnp.einsum("ab,nbc->nac", Ad_inv_dot, J_base)
                    ).at[i].set(Ad_inv @ Td)
                return (
                    g_ij,
                    self._final_size_jacobian(J_ij),
                    self._final_size_jacobian(Jd_ij),
                )

            return vmap(point_kinematics)(s_local_i)

        g_ps, J_ps, Jd_ps = vmap(segment_kinematics)(
            jnp.arange(self.num_segments, dtype=jnp.int32),
            xi,
            xid,
            g_bases,
            J_bases,
            Jd_bases,
            s_local,
        )
        if return_tips:
            return Ws_scaled, g_ps, J_ps, Jd_ps, g_tips, J_tips, Jd_tips
        return Ws_scaled, g_ps, J_ps, Jd_ps

    def _integration_kinematics(
        self, q: Array, qd: Array, convective_only_jd: bool = False
    ) -> tuple[Array, Array, Array, Array]:
        Ws_scaled, g_ps, J_full, Jd_full = self._integration_kinematics_full(
            q, qd, convective_only_jd
        )
        return (
            Ws_scaled,
            g_ps,
            jnp.einsum("sqab,bd->sqad", J_full, self.B_xi),
            jnp.einsum("sqab,bd->sqad", Jd_full, self.B_xi),
        )

    def integration_kinematics(
        self, q: Array, qd: Array
    ) -> tuple[Array, Array, Array]:
        _, g_ps, J_ps, Jd_ps = self._integration_kinematics(q, qd)
        return g_ps, J_ps, Jd_ps

    def _mass_properties_kinematics(
        self,
        q: Array,
        qd: Array,
        convective_only_jd: bool = False,
        full: bool = False,
    ) -> tuple[Array, ...]:
        (
            Ws,
            g_ps,
            J_full,
            Jd_full,
            g_tips,
            J_tips,
            Jd_tips,
        ) = self._integration_kinematics_full(
            q, qd, convective_only_jd, return_tips=True
        )
        basis = (
            jnp.eye(self.num_strains, dtype=J_full.dtype)
            if full
            else self.B_xi
        )
        J_ps = jnp.einsum("sqab,bd->sqad", J_full, basis)
        Jd_ps = jnp.einsum("sqab,bd->sqad", Jd_full, basis)
        rod_transforms = self.rod_transforms
        rod_adjoint_inverses = self.rod_adjoint_inverses
        J_rods = jnp.einsum("srab,sqbd->sqrad", rod_adjoint_inverses, J_ps)
        Jd_rods = jnp.einsum("srab,sqbd->sqrad", rod_adjoint_inverses, Jd_ps)
        g_rods = jnp.einsum("sqab,srbc->sqrac", g_ps, rod_transforms)
        rod_mass_matrices = self.rod_mass_matrices
        masses = self.segment_masses
        inertias = self.segment_inertias
        cog_transforms = self.cog_transforms
        cog_adjoint_inverses = self.cog_adjoint_inverses
        tip_indices = jnp.arange(self.num_segments, dtype=jnp.int32)
        J_tips = J_tips[tip_indices, tip_indices]
        Jd_tips = Jd_tips[tip_indices, tip_indices]
        J_tips = jnp.einsum("sab,bd->sad", J_tips, basis)
        Jd_tips = jnp.einsum("sab,bd->sad", Jd_tips, basis)
        J_cogs = jnp.einsum("sab,sbd->sad", cog_adjoint_inverses, J_tips)
        Jd_cogs = jnp.einsum("sab,sbd->sad", cog_adjoint_inverses, Jd_tips)
        g_cogs = jnp.einsum("sab,sbc->sac", g_tips, cog_transforms)

        payload_transform = self.payload_transform
        payload_adjoint_inverse = self.payload_adjoint_inverse
        J_payload = payload_adjoint_inverse @ J_tips[-1]
        Jd_payload = payload_adjoint_inverse @ Jd_tips[-1]
        g_payload = g_tips[-1] @ payload_transform
        payload_mass_matrix = self.payload_mass_matrix
        return (
            Ws,
            g_rods,
            J_rods,
            Jd_rods,
            rod_mass_matrices,
            g_cogs,
            J_cogs,
            Jd_cogs,
            masses,
            inertias,
            J_payload,
            Jd_payload,
            g_payload,
            payload_mass_matrix,
        )

    def _assemble_dynamics_terms(
        self,
        q: Array,
        qd: Array,
        *,
        convective_only_jd: bool = False,
        full: bool = False,
        return_matrix: bool = False,
    ) -> tuple[Array, Array, Array]:
        (
            Ws,
            g_rods,
            J_rods,
            Jd_rods,
            rod_mass_matrices,
            g_cogs,
            J_cogs,
            Jd_cogs,
            masses,
            inertias,
            J_payload,
            Jd_payload,
            g_payload,
            payload_mass_matrix,
        ) = self._mass_properties_kinematics(
            q, qd, convective_only_jd, full=full
        )
        velocity = self.B_xi @ qd if full else qd

        rod_eta = jnp.einsum("sqrad,d->sqra", J_rods, velocity)
        rod_coad = vmap(vmap(vmap(se2.coadjoint)))(rod_eta)
        rod_g_inv = vmap(vmap(vmap(se2.adjoint_inverse)))(g_rods)
        rod_mj = jnp.einsum("srab,sqrbd->sqrad", rod_mass_matrices, J_rods)
        rod_mjd = jnp.einsum("srab,sqrbd->sqrad", rod_mass_matrices, Jd_rods)
        rod_local_gravity = jnp.einsum(
            "sqrab,b->sqra", rod_g_inv, self.g
        )
        rod_gravity_wrench = jnp.einsum(
            "srab,sqrb->sqra", rod_mass_matrices, rod_local_gravity
        )
        B_rods = jnp.einsum(
            "sq,sqrad,srab,sqrbe->de",
            Ws,
            J_rods,
            rod_mass_matrices,
            J_rods,
        )
        C_rods = jnp.einsum(
            "sq,sqrad,sqrae->de", Ws, J_rods, rod_mjd + jnp.einsum(
                "sqrae,sqred->sqrad", rod_coad, rod_mj
            )
        )
        G_rods = -jnp.einsum("sq,sqrad,sqra->d", Ws, J_rods, rod_gravity_wrench)

        cog_matrices = jnp.zeros((*masses.shape, 3, 3), dtype=Ws.dtype)
        cog_matrices = cog_matrices.at[:, 0, 0].set(inertias)
        cog_matrices = cog_matrices.at[:, 1, 1].set(masses)
        cog_matrices = cog_matrices.at[:, 2, 2].set(masses)
        cog_velocity = jnp.einsum("sad,d->sa", J_cogs, velocity)
        cog_coad = vmap(se2.coadjoint)(cog_velocity)
        cog_mj = jnp.einsum("sab,sbd->sad", cog_matrices, J_cogs)
        cog_mjd = jnp.einsum("sab,sbd->sad", cog_matrices, Jd_cogs)
        cog_g_inv = vmap(se2.adjoint_inverse)(g_cogs)
        cog_local_gravity = jnp.einsum("sab,b->sa", cog_g_inv, self.g)
        cog_gravity_wrench = jnp.einsum(
            "sab,sb->sa", cog_matrices, cog_local_gravity
        )
        B_cogs = jnp.einsum("sad,sab,sbe->de", J_cogs, cog_matrices, J_cogs)
        C_cogs = jnp.einsum("sad,sae->de", J_cogs, cog_mjd + jnp.einsum(
            "sab,sbe->sae", cog_coad, cog_mj
        ))
        G_cogs = -jnp.einsum("sad,sa->d", J_cogs, cog_gravity_wrench)

        payload_velocity = J_payload @ velocity
        payload_coad = se2.coadjoint(payload_velocity)
        payload_mj = payload_mass_matrix @ J_payload
        payload_mjd = payload_mass_matrix @ Jd_payload
        payload_gravity_wrench = payload_mass_matrix @ (
            se2.adjoint_inverse(g_payload) @ self.g
        )
        B_payload = J_payload.T @ payload_mass_matrix @ J_payload
        C_payload = J_payload.T @ (payload_mjd + payload_coad @ payload_mj)
        G_payload = -J_payload.T @ payload_gravity_wrench

        B = B_rods + B_cogs + B_payload
        C = C_rods + C_cogs + C_payload
        return (
            B,
            C if return_matrix else C @ velocity,
            G_rods + G_cogs + G_payload,
        )

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

    def _gravitational_energy(self, q: Array) -> Array:
        return self._gravitational_energy_full_from_xi(self.strain(q))

    def _assemble_gravity_force(self, q: Array, *, full: bool = False) -> Array:
        """Assemble gravity from pose/Jacobian data only."""
        Ws, g_ps, J_full, g_tips, J_tips = self._integration_kinematics_gravity(q)
        basis = (
            jnp.eye(self.num_strains, dtype=J_full.dtype)
            if full
            else self.B_xi
        )
        J_ps = jnp.einsum("sqab,bd->sqad", J_full, basis)
        rod_transforms = self.rod_transforms
        rod_adjoint_inverses = self.rod_adjoint_inverses
        J_rods = jnp.einsum("srab,sqbd->sqrad", rod_adjoint_inverses, J_ps)
        g_rods = jnp.einsum("sqab,srbc->sqrac", g_ps, rod_transforms)
        rod_local_gravity = jnp.einsum(
            "sqrab,b->sqra",
            vmap(vmap(vmap(se2.adjoint_inverse)))(g_rods),
            self.g,
        )
        rod_gravity_wrench = jnp.einsum(
            "srab,sqrb->sqra", self.rod_mass_matrices, rod_local_gravity
        )
        G_rods = -jnp.einsum(
            "sq,sqrad,sqra->d", Ws, J_rods, rod_gravity_wrench
        )

        tip_indices = jnp.arange(self.num_segments, dtype=jnp.int32)
        J_tips = J_tips[tip_indices, tip_indices]
        J_tips = jnp.einsum("sab,bd->sad", J_tips, basis)
        J_cogs = jnp.einsum(
            "sab,sbd->sad", self.cog_adjoint_inverses, J_tips
        )
        g_cogs = jnp.einsum("sab,sbc->sac", g_tips, self.cog_transforms)
        cog_matrices = jnp.zeros(
            (*self.segment_masses.shape, 3, 3), dtype=Ws.dtype
        )
        cog_matrices = cog_matrices.at[:, 0, 0].set(self.segment_inertias)
        cog_matrices = cog_matrices.at[:, 1, 1].set(self.segment_masses)
        cog_matrices = cog_matrices.at[:, 2, 2].set(self.segment_masses)
        cog_gravity_wrench = jnp.einsum(
            "sab,sb->sa",
            cog_matrices,
            jnp.einsum(
                "sab,b->sa",
                vmap(se2.adjoint_inverse)(g_cogs),
                self.g,
            ),
        )
        G_cogs = -jnp.einsum("sad,sa->d", J_cogs, cog_gravity_wrench)

        J_payload = self.payload_adjoint_inverse @ J_tips[-1]
        g_payload = g_tips[-1] @ self.payload_transform
        payload_gravity_wrench = self.payload_mass_matrix @ (
            se2.adjoint_inverse(g_payload) @ self.g
        )
        G_payload = -J_payload.T @ payload_gravity_wrench
        return G_rods + G_cogs + G_payload

    def _inertia_full_matrix(self, q: Array) -> Array:
        return self._assemble_dynamics_terms(
            q, jnp.zeros((self.num_dofs,), dtype=q.dtype), full=True
        )[0]

    def inertia_matrix(self, q: Array) -> Array:
        return self._assemble_dynamics_terms(
            q, jnp.zeros((self.num_dofs,), dtype=q.dtype)
        )[0]

    def _inertia_derivative_matrix(self, q: Array) -> Array:
        """Compute ``dB/dq`` from the analytical Jacobian derivatives."""
        directions = jnp.eye(self.num_dofs, dtype=q.dtype)
        geometry = self._integration_geometry_full(q)
        (
            Ws,
            _,
            J_rods,
            rod_mass_matrices,
            _,
            J_cogs,
            masses,
            inertias,
            J_payload,
            _,
            payload_mass_matrix,
        ) = self._mass_geometry_from_integration(geometry, full=False)
        velocity_basis = self.B_xi

        def directional_derivative(direction: Array) -> Array:
            Jd_full, Jd_tips = (
                self._integration_jacobian_time_derivative_from_geometry(
                    direction, geometry
                )
            )
            Jd_ps = jnp.einsum("sqab,bd->sqad", Jd_full, velocity_basis)
            Jd_rods = jnp.einsum(
                "srab,sqbd->sqrad", self.rod_adjoint_inverses, Jd_ps
            )
            tip_indices = jnp.arange(self.num_segments, dtype=jnp.int32)
            Jd_tips = Jd_tips[tip_indices, tip_indices]
            Jd_tips = jnp.einsum("sab,bd->sad", Jd_tips, velocity_basis)
            Jd_cogs = jnp.einsum(
                "sab,sbd->sad", self.cog_adjoint_inverses, Jd_tips
            )
            Jd_payload = self.payload_adjoint_inverse @ Jd_tips[-1]
            rod_term = jnp.einsum(
                "sq,sqrad,srab,sqrbe->de",
                Ws,
                Jd_rods,
                rod_mass_matrices,
                J_rods,
            ) + jnp.einsum(
                "sq,sqrad,srab,sqrbe->de",
                Ws,
                J_rods,
                rod_mass_matrices,
                Jd_rods,
            )
            cog_matrices = jnp.zeros((*masses.shape, 3, 3), dtype=q.dtype)
            cog_matrices = cog_matrices.at[:, 0, 0].set(inertias)
            cog_matrices = cog_matrices.at[:, 1, 1].set(masses)
            cog_matrices = cog_matrices.at[:, 2, 2].set(masses)
            cog_term = jnp.einsum(
                "sad,sab,sbe->de", Jd_cogs, cog_matrices, J_cogs
            ) + jnp.einsum("sad,sab,sbe->de", J_cogs, cog_matrices, Jd_cogs)
            payload_term = Jd_payload.T @ payload_mass_matrix @ J_payload + (
                J_payload.T @ payload_mass_matrix @ Jd_payload
            )
            return rod_term + cog_term + payload_term

        return jnp.moveaxis(vmap(directional_derivative)(directions), 0, -1)

    def _coriolis_full_matrix(self, q: Array, qd: Array) -> Array:
        return self._assemble_dynamics_terms(
            q, qd, full=True, return_matrix=True
        )[1]

    def coriolis_matrix(self, q: Array, qd: Array) -> Array:
        dB = self._inertia_derivative_matrix(q)
        return 0.5 * jnp.einsum(
            "abk,k->ab",
            dB
            + jnp.swapaxes(dB, 1, 2)
            - jnp.transpose(dB, (2, 1, 0)),
            qd,
        )

    def _gravitational_full_force(self, q: Array) -> Array:
        return self._assemble_gravity_force(q, full=True)

    def _gravitational_force(self, q: Array) -> Array:
        return self._assemble_gravity_force(q)

    def _build_nominal_stiffness_full(self, mapping: Array) -> Array:
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
        blocks = jnp.einsum("...ji,...jk,...kl->...il", mapping, Shat_rod, mapping)
        return blk_diag(jnp.sum(blocks, axis=1))

    def _nominal_stiffness_full(self) -> Array:
        """Return the cached full HSA stiffness matrix."""
        return self.K_full

    def Shat(self) -> Array:
        return self.K_full

    def stiffness_matrix(self) -> Array:
        return self.K_active

    def _stiffness_full_vector(self, q: Array) -> Array:
        return self.Shat() @ (self.strain(q) - self.ref_strains())

    def elastic_force(self, q: Array) -> Array:
        return self.B_xi.T @ self._stiffness_full_vector(q)

    def _elastic_energy(self, q: Array) -> Array:
        return 0.5 * q @ self.stiffness_matrix() @ q

    def _build_damping_full(self, mapping: Array) -> Array:
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
        blocks = jnp.einsum(
            "...ji,...jk,...kl->...il", mapping, damping_rod, mapping
        )
        return blk_diag(jnp.sum(blocks, axis=1))

    def _damping_full_matrix(self) -> Array:
        """Return the cached full HSA damping matrix."""
        return self.D_full

    def damping_matrix(self, q: Array) -> Array:
        del q
        return self.D_active

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
        return self._assemble_dynamics_terms(
            q, qd, convective_only_jd=True
        )

    @eqx.filter_jit
    def forward_dynamics(
        self, t: Array, y: Array, actuation_args: tuple | None = None
    ) -> Array:
        """Compute the HSA state derivative."""
        if actuation_args is None:
            u, tau_ext = None, None
        elif len(actuation_args) == 1:
            u, tau_ext = actuation_args[0], None
        elif len(actuation_args) == 2:
            u, tau_ext = actuation_args
        else:
            raise ValueError("actuation_args must be a tuple of length 1 or 2.")
        if u is None:
            u = jnp.zeros((self.num_actuators,), dtype=y.dtype)
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

        tau_u = self.actuation_force(q, u, qd=qd)
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

