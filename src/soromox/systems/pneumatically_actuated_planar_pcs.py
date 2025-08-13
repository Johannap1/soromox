from jax import Array, vmap
import jax.numpy as jnp
from typing import Dict, Optional

import equinox as eqx

from .planar_pcs import PlanarPCS


class PneumaticallyActuatedPlanarPCS(PlanarPCS):
    r_chamber_in: Array  # inner radius of each segment's chamber, shape (num_segments,)
    r_chamber_out: (
        Array  # outer radius of each segment's chamber, shape (num_segments,)
    )
    phi_chamber: Array  # sector angle of each segment's chamber, shape (num_segments,)
    num_chambers: int = eqx.field(
        static=True, default=4
    )  # number of pneumatic chambers per segment

    actuation_basis: Array  # actuation basis, shape (num_segments * 2, num_actuators)

    simplified_actuation_mapping: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        num_segments: int,
        params: Dict[str, Array],
        order_gauss: int = 5,
        strain_selector: Optional[Array] = None,
        xi_ref: Optional[Array] = None,
        segment_actuation_selector: Optional[Array] = None,
        simplified_actuation_mapping: bool = False,
    ):
        super().__init__(
            num_segments=num_segments,
            params=params,
            order_gauss=order_gauss,
            strain_selector=strain_selector,
            xi_ref=xi_ref,
        )

        if segment_actuation_selector is None:
            segment_actuation_selector = jnp.ones(num_segments, dtype=bool)

        # self.segment_indices_to_actuate = jnp.array(
        #     [i for i, act in enumerate(segment_actuation_selector) if act]
        # )

        self.num_actuators = (
            int(jnp.sum(segment_actuation_selector)) * 2
        )  # each segment has two tendons

        actuation_basis = jnp.zeros((2 * self.num_segments, self.num_actuators))
        actuation_basis_cumsum = jnp.cumsum(segment_actuation_selector)
        for i in range(self.num_segments):
            j = int(actuation_basis_cumsum[i].item()) - 1
            if segment_actuation_selector[i].item() is True:
                actuation_basis = actuation_basis.at[2 * i, j].set(1.0)
                actuation_basis = actuation_basis.at[2 * i + 1, j + 1].set(1.0)
        self.actuation_basis = actuation_basis

        self.simplified_actuation_mapping = simplified_actuation_mapping

        self._set_params(params)

    def _set_params(self, params: Dict[str, Array]):
        """
        Set the parameters of the tendon-driven planar PCS.

        Args:
            params (Dict[str, Array]): Dictionary containing the parameters of the robot.
                Dictionary containing the robot parameters:
                - "th0": (optional) float
                    Initial orientation angle [rad]
                    Default is 90 degrees (1.57 radians).
                - "L": List/Array of num_segments floats
                    Length of each segment [m]
                - "r": List/Array of num_segments floats
                    Radius of each segment [m]
                - "rho": List/Array of num_segments floats
                    Density of each segment [kg/m^3]
                - "g": List/Array of 2 floats [gx, gy]
                    Gravitational acceleration vector [m/s^2]
                - "E": List/Array of num_segments floats
                    Elastic modulus of each segment [Pa]
                - "G": List/Array of num_segments floats
                    Shear modulus of each segment [Pa]
                - "D": List/Array of (num_segments x num_segments) floats
                    Damping matrix of each segment [Pa*s]
                - "r_chamber_in" : Array of num_segments floats
                    Inner radius of each segment's pneumatic chamber [m]
                - "r_chamber_out" : Array of num_segments floats
                    Outer radius of each segment's pneumatic chamber [m]
                -"phi_chamber" : Array of num_segments floats
                    Sector angle of each segment's pneumatic chamber [rad]

        """
        super()._set_params(params)

        # Chamfer parameters
        try:
            r_chamber_in = params["r_chamber_in"]
        except KeyError:
            raise KeyError(
                "The parameter 'r_chamber_in' (inner radius of each segment's pneumatic chamber) is required for the pneumatically actuated planar PCS."
            )
        if not isinstance(r_chamber_in, (list, jnp.ndarray)):
            raise TypeError(
                "The parameter 'r_chamber_in' must be a list or a jnp.ndarray."
            )
        if len(r_chamber_in) != self.num_segments:
            raise ValueError(
                f"The parameter 'r_chamber_in' must have the same length as the number of segments ({self.num_segments})."
            )
        self.r_chamber_in = jnp.asarray(r_chamber_in, dtype=jnp.float64)
        try:
            r_chamber_out = params["r_chamber_out"]
        except KeyError:
            raise KeyError(
                "The parameter 'r_chamber_out' (outer radius of each segment's pneumatic chamber) is required for the pneumatically actuated planar PCS."
            )
        if not isinstance(r_chamber_out, (list, jnp.ndarray)):
            raise TypeError(
                "The parameter 'r_chamber_out' must be a list or a jnp.ndarray."
            )
        if len(r_chamber_out) != self.num_segments:
            raise ValueError(
                f"The parameter 'r_chamber_out' must have the same length as the number of segments ({self.num_segments})."
            )
        self.r_chamber_out = jnp.asarray(r_chamber_out, dtype=jnp.float64)
        try:
            phi_chamber = params["phi_chamber"]
        except KeyError:
            raise KeyError(
                "The parameter 'phi_chamber' (sector angle of each segment's pneumatic chamber) is required for the pneumatically actuated planar PCS."
            )
        if not isinstance(phi_chamber, (list, jnp.ndarray)):
            raise TypeError(
                "The parameter 'phi_chamber' must be a list or a jnp.ndarray."
            )
        if len(phi_chamber) != self.num_segments:
            raise ValueError(
                f"The parameter 'phi_chamber' must have the same length as the number of segments ({self.num_segments})."
            )
        self.phi_chamber = jnp.asarray(phi_chamber, dtype=jnp.float64)

    def update_params(
        self, params: Dict[str, Array]
    ) -> "PneumaticallyActuatedPlanarPCS":
        """
        Update the parameters of the tendon-driven planar PCS.

        Args:
            params (Dict[str, Array]):
                Dictionary that contains the robot parameters to update:
                - "th0": (optional) float
                    Initial orientation angle [rad]
                - "L": List/Array of num_segments floats
                    Length of each segment [m]
                - "r": List/Array of num_segments floats
                    Radius of each segment [m]
                - "rho": List/Array of num_segments floats
                    Density of each segment [kg/m^3]
                - "g": List/Array of 2 floats [gx, gy]
                    Gravitational acceleration vector [m/s^2]
                - "E": List/Array of num_segments floats
                    Elastic modulus of each segment [Pa]
                - "G": List/Array of num_segments floats
                    Shear modulus of each segment [Pa]
                - "D": List/Array of (num_segments x num_segments) floats
                    Damping matrix of each segment [Pa*s]
                - "r_chamber_in" : Array of num_segments floats
                    Inner radius of each segment's pneumatic chamber [m]
                - "r_chamber_out" : Array of num_segments floats
                    Outer radius of each segment's pneumatic chamber [m]
                -"phi_chamber" : Array of num_segments floats
                    Sector angle of each segment's pneumatic chamber [rad]
        Returns:
            updated_self (PneumaticallyActuatedPlanarPCS):
                A new instance of PneumaticallyActuatedPlanarPCS with updated parameters.
        """
        # Apply updates sequentially
        updated_self = super().update_params(params)

        if "r_chamber_in" in params:
            r_chamber_in = params["r_chamber_in"]
            if not isinstance(r_chamber_in, (list, jnp.ndarray)):
                raise TypeError(
                    "The parameter 'r_chamber_in' must be a list or a jnp.ndarray."
                )
            if len(r_chamber_in) != self.num_segments:
                raise ValueError(
                    f"The parameter 'r_chamber_in' must have the same length as the number of segments ({self.num_segments})."
                )
            updated_self = eqx.tree_at(
                lambda x: x.r_chamber_in,
                updated_self,
                jnp.asarray(r_chamber_in, dtype=jnp.float64),
            )

        if "r_chamber_out" in params:
            r_chamber_out = params["r_chamber_out"]
            if not isinstance(r_chamber_out, (list, jnp.ndarray)):
                raise TypeError(
                    "The parameter 'r_chamber_out' must be a list or a jnp.ndarray."
                )
            if len(r_chamber_out) != self.num_segments:
                raise ValueError(
                    f"The parameter 'r_chamber_out' must have the same length as the number of segments ({self.num_segments})."
                )
            updated_self = eqx.tree_at(
                lambda x: x.r_chamber_out,
                updated_self,
                jnp.asarray(r_chamber_out, dtype=jnp.float64),
            )

        if "phi_chamber" in params:
            phi_chamber = params["phi_chamber"]
            if not isinstance(phi_chamber, (list, jnp.ndarray)):
                raise TypeError(
                    "The parameter 'phi_chamber' must be a list or a jnp.ndarray."
                )
            if len(phi_chamber) != self.num_segments:
                raise ValueError(
                    f"The parameter 'phi_chamber' must have the same length as the number of segments ({self.num_segments})."
                )
            updated_self = eqx.tree_at(
                lambda x: x.phi_chamber,
                updated_self,
                jnp.asarray(phi_chamber, dtype=jnp.float64),
            )

        return updated_self

    @eqx.filter_jit
    def _local_chamber_cross_sectional_area(self, i: int) -> Array:
        """
        Compute the local cross-sectional area of one pneumatic chamber for the i-th segment.

        Args:
            i (int): index of the segment

        Returns:
            A_one_chamber_i (Array): local cross-sectional area of one pneumatic chamber of the i-th segment
        """
        A_one_chamber_i = (
            self.phi_chamber[i]
            / 2
            * (self.r_chamber_out[i] ** 2 - self.r_chamber_in[i] ** 2)
        )

        return A_one_chamber_i

    @eqx.filter_jit
    def _local_cross_sectional_area(self, i: int) -> Array:
        """
        Compute the local cross-sectional area for the i-th segment.

        Args:
            i (int): index of the segment

        Returns:
            A_i (Array): local cross-sectional area of the i-th segment
        """
        A_full_i = super()._local_cross_sectional_area(
            i
        )  # Full cross-sectional area of the i-th segment without chambers
        A_one_chamber_i = self._local_chamber_cross_sectional_area(i)
        A_i = (
            A_full_i - self.num_chambers * A_one_chamber_i
        )  # Subtract the area of the four chambers

        return A_i

    @eqx.filter_jit
    def _local_chamber_second_moment_of_area(self, i: int) -> Array:
        """
        Compute the local second moment of area of one pneumatic chamber for the i-th segment.

        Args:
            i (int): index of the segment

        Returns:
            I_one_chamber_i (Array): local second moment of area of one pneumatic chamber of the i-th segment
        """
        I_one_chamber_i = (
            (self.phi_chamber[i] - jnp.sin(self.phi_chamber[i]))
            / 8
            * (self.r_chamber_out[i] ** 4 - self.r_chamber_in[i] ** 4)
        )

        return I_one_chamber_i

    @eqx.filter_jit
    def _local_second_moment_of_area(self, i: int) -> Array:
        """
        Compute the local second moment of area for the i-th segment.

        Args:
            i (int): index of the segment

        Returns:
            I_i (Array): local second moment of area of the i-th segment
        """
        I_full_i = super()._local_second_moment_of_area(
            i
        )  # Full second moment of area of the i-th segment without chambers
        I_one_chamber_i = self._local_chamber_second_moment_of_area(i)
        I_i = (
            I_full_i - self.num_chambers * I_one_chamber_i
        )  # Subtract the second moment of area of the four chambers

        return I_i

    # @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Compute the actuation matrix of the robot.
        We assume that each segment contains four identical and symmetric pneumatic chambers with pressures
        p1, p2, p3, and p4, where:
            - p1 and p3 are the right and left chamber pressures respectively,
            - p2 and p4 are the back and front chamber pressures respectively.
        The front and back chambers do not exert a level arm (i.e., a bending moment) on the segment.
        We map the control inputs u1 and u2 as follows to the pressures:
            p1 = u1 (right chamber)
            p2 = (u1 + u2) / 2
            p3 = u2 (left chamber)
            p4 = (u1 + u2) / 2

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            A (Array): Actuation matrix of shape (num_active_strains, num_actuators)
        """

        def A_segment_i(i: int) -> Array:
            # Area of one pneumatic chamber
            A_one_chamber = self._local_chamber_cross_sectional_area(i)

            # Distance from the center of the segment to the center of pressure of one chamber
            r_center_of_pressure = (
                2
                / 3
                * jnp.sinc(self.phi_chamber[i] / 2)
                * (self.r_chamber_out[i] ** 3 - self.r_chamber_in[i] ** 3)
                / (self.r_chamber_out[i] ** 2 - self.r_chamber_in[i] ** 2)
            )

            if self.simplified_actuation_mapping:
                A_full_segment_i = jnp.array(
                    [
                        [
                            A_one_chamber * r_center_of_pressure,
                            -A_one_chamber * r_center_of_pressure,
                        ],  # actuation on the bending
                        [
                            2 * A_one_chamber,
                            2 * A_one_chamber,
                        ],  # actuation on the axial strain
                        [0.0, 0.0],  # actuation on the shear strain
                    ]
                )

                A_segment_i = self.B_xi.T @ A_full_segment_i
            else:
                # Jacobians at the base and tip of the segment
                J_base_i = self.jacobian_bodyframe(q, self.L_cum[i])
                J_tip_i = self.jacobian_bodyframe(q, self.L_cum[i + 1])

                # Contribution of the proximal end
                A_segment_i_proximal_end = J_base_i.T @ jnp.array(
                    [
                        [
                            -A_one_chamber * r_center_of_pressure,
                            A_one_chamber * r_center_of_pressure,
                        ],
                        [-2 * A_one_chamber, -2 * A_one_chamber],
                        [0.0, 0.0],
                    ]
                )
                # Contribution of the distal end
                A_segment_i_distal_end = J_tip_i.T @ jnp.array(
                    [
                        [
                            A_one_chamber * r_center_of_pressure,
                            -A_one_chamber * r_center_of_pressure,
                        ],
                        [2 * A_one_chamber, 2 * A_one_chamber],
                        [0.0, 0.0],
                    ]
                )

                # sum the contributions of the distal and proximal ends
                A_segment_i = A_segment_i_distal_end + A_segment_i_proximal_end

            return A_segment_i

        A_blocks_tot = vmap(A_segment_i)(
            jnp.arange(self.num_segments),
        )

        # # For debugging purposes, we can use a for loop instead of vmap
        # A_blocks_tot = jnp.stack(
        #     [A_segment_i(i) for i in range(self.num_segments)],
        #     axis=0
        # )

        # we need to sum the contributions of the actuation of each segment
        A = jnp.concatenate(A_blocks_tot, axis=-1)

        # apply the actuation_basis
        A = A @ self.actuation_basis

        return A
