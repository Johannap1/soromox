from jax import Array, vmap
import jax.numpy as jnp
from typing import Dict, Optional

import equinox as eqx

from .planar_pcs import PlanarPCS


class PneumaticallyActuatedPlanarPCS(PlanarPCS):
    
    r_cham_in : Array  # inner radius of each segment's chamfer, shape (num_segments,)
    r_cham_out : Array  # outer radius of each segment's chamfer, shape (num_segments,)
    varphi_cham : Array  # sector angle of each segment's chamfer, shape (num_segments,)
    
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
                - "r_cham_in" : Array of num_segments floats
                    Inner radius of each segment's chamfer [m]
                - "r_cham_out" : Array of num_segments floats
                    Outer radius of each segment's chamfer [m]
                -"varphi_cham" : Array of num_segments floats
                    Sector angle of each segment's chamfer [rad]
    
        """
        super()._set_params(params)
        
        # Chamfer parameters
        try:
            r_cham_in = params["r_cham_in"]
        except KeyError:
            raise KeyError(
                "The parameter 'r_cham_in' (inner radius of each segment's chamfer) is required for the pneumatically actuated planar PCS."
            )
        if not isinstance(r_cham_in, (list, jnp.ndarray)):
            raise TypeError("The parameter 'r_cham_in' must be a list or a jnp.ndarray.")
        if len(r_cham_in) != self.num_segments:
            raise ValueError(
                f"The parameter 'r_cham_in' must have the same length as the number of segments ({self.num_segments})."
            )
        self.r_cham_in = jnp.asarray(r_cham_in, dtype=jnp.float64)
        try:
            r_cham_out = params["r_cham_out"]
        except KeyError:
            raise KeyError(
                "The parameter 'r_cham_out' (outer radius of each segment's chamfer) is required for the pneumatically actuated planar PCS."
            )
        if not isinstance(r_cham_out, (list, jnp.ndarray)):
            raise TypeError("The parameter 'r_cham_out' must be a list or a jnp.ndarray.")
        if len(r_cham_out) != self.num_segments:
            raise ValueError(
                f"The parameter 'r_cham_out' must have the same length as the number of segments ({self.num_segments})."
            )
        self.r_cham_out = jnp.asarray(r_cham_out, dtype=jnp.float64)
        try:
            varphi_cham = params["varphi_cham"]
        except KeyError:
            raise KeyError(
                "The parameter 'varphi_cham' (sector angle of each segment's chamfer) is required for the pneumatically actuated planar PCS."
            )
        if not isinstance(varphi_cham, (list, jnp.ndarray)):
            raise TypeError("The parameter 'varphi_cham' must be a list or a jnp.ndarray.")
        if len(varphi_cham) != self.num_segments:
            raise ValueError(
                f"The parameter 'varphi_cham' must have the same length as the number of segments ({self.num_segments})."
            )
        self.varphi_cham = jnp.asarray(varphi_cham, dtype=jnp.float64)

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
                - "r_cham_in" : Array of num_segments floats
                    Inner radius of each segment's chamfer [m]
                - "r_cham_out" : Array of num_segments floats
                    Outer radius of each segment's chamfer [m]
                -"varphi_cham" : Array of num_segments floats
                    Sector angle of each segment's chamfer [rad]
        Returns:
            updated_self (PneumaticallyActuatedPlanarPCS):
                A new instance of PneumaticallyActuatedPlanarPCS with updated parameters.
        """
        # Apply updates sequentially
        updated_self = super().update_params(params)

        if "r_cham_in" in params:
            r_cham_in = params["r_cham_in"]
            if not isinstance(r_cham_in, (list, jnp.ndarray)):
                raise TypeError("The parameter 'r_cham_in' must be a list or a jnp.ndarray.")
            if len(r_cham_in) != self.num_segments:
                raise ValueError(
                    f"The parameter 'r_cham_in' must have the same length as the number of segments ({self.num_segments})."
                )
            updated_self = eqx.tree_at(
                lambda x: x.r_cham_in, updated_self, jnp.asarray(r_cham_in, dtype=jnp.float64)
            )
        
        if "r_cham_out" in params:
            r_cham_out = params["r_cham_out"]
            if not isinstance(r_cham_out, (list, jnp.ndarray)):
                raise TypeError("The parameter 'r_cham_out' must be a list or a jnp.ndarray.")
            if len(r_cham_out) != self.num_segments:
                raise ValueError(
                    f"The parameter 'r_cham_out' must have the same length as the number of segments ({self.num_segments})."
                )
            updated_self = eqx.tree_at(
                lambda x: x.r_cham_out, updated_self, jnp.asarray(r_cham_out, dtype=jnp.float64)
            )
        
        if "varphi_cham" in params:
            varphi_cham = params["varphi_cham"]
            if not isinstance(varphi_cham, (list, jnp.ndarray)):
                raise TypeError("The parameter 'varphi_cham' must be a list or a jnp.ndarray.")
            if len(varphi_cham) != self.num_segments:
                raise ValueError(
                    f"The parameter 'varphi_cham' must have the same length as the number of segments ({self.num_segments})."
                )
            updated_self = eqx.tree_at(
                lambda x: x.varphi_cham, updated_self, jnp.asarray(varphi_cham, dtype=jnp.float64)
            )
        
        return updated_self

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Compute the actuation matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            A (Array): Actuation matrix of shape (num_active_strains, num_actuators)
        """
        
        chi_tips_list = vmap(
            lambda s: self.forward_kinematics(q, s)
        )(self.L_cum)
        
        J_tips_list = vmap(
            lambda s: self.jacobian(q, s)
        )(self.L_cum)
        
        def compute_actuation_matrix_for_segment(
            r_cham_in: Array,
            r_cham_out: Array,
            varphi_cham: Array,
            chi_pe: Array,
            chi_de: Array,
            J_pe: Array,
            J_de: Array,
        ) -> Array:
            """
            Compute the actuation matrix for a single segment.
            We assume that each segment contains four identical and symmetric pneumatic chambers with pressures
            p1, p2, p3, and p4, where p1 and p3 are the right and left chamber pressures respectively, and
            p2 and p4 are the back and front chamber pressures respectively. The front and back chambers
            do not exert a level arm (i.e., a bending moment) on the segment.
            We map the control inputs u1 and u2 as follows to the pressures:
                p1 = u1 (right chamber)
                p2 = (u1 + u2) / 2
                p3 = u2 (left chamber)
                p4 = (u1 + u2) / 2

            Args:
                r_cham_in: inner radius of each segment chamber
                r_cham_out: outer radius of each segment chamber
                varphi_cham: sector angle of each segment chamber
                chi_pe: pose of the proximal end (i.e., the base) of the segment as array of shape (3,)
                chi_de: pose of the distal end (i.e., the tip) of the segment as array of shape (3,)
                J_pe: Jacobian of the proximal end of the segment as array of shape (3, n_q)
                J_de: Jacobian of the distal end of the segment as array of shape (3, n_q)
            Returns:
                A_sm: actuation matrix of shape (n_xi, 2)
            """
            # orientation of the proximal and distal ends of the segment
            th_pe, th_de = chi_pe[0], chi_de[0]

            # compute the area of each pneumatic chamber (we assume identical chambers within a segment)
            A_cham = 0.5 * varphi_cham * (r_cham_out**2 - r_cham_in**2)
            # compute the center of pressure of the pneumatic chamber
            r_cop = (
                2
                / 3
                * jnp.sinc(0.5 * varphi_cham)
                * (r_cham_out**3 - r_cham_in**3)
                / (r_cham_out**2 - r_cham_in**2)
            )

            if self.simplified_actuation_mapping:
                A_sm = self.B_xi.T @ jnp.array(
                    [
                        [A_cham * r_cop, -A_cham * r_cop],
                        # [0.0, 0.0],
                        [2 * A_cham, 2 * A_cham],
                        [0.0, 0.0],
                    ]
                )
            else:
                # compute the actuation matrix that collects the contributions of the pneumatic chambers in the given segment
                # first we consider the contribution of the distal end
                A_sm_de = J_de.T @ jnp.array(
                    [
                        [-2 * A_cham * jnp.sin(th_de), -2 * A_cham * jnp.sin(th_de)],
                        # [2 * A_cham * jnp.cos(th_de), 2 * A_cham * jnp.cos(th_de)],
                        [A_cham * r_cop, -A_cham * r_cop],
                        [2 * A_cham * jnp.cos(th_de), 2 * A_cham * jnp.cos(th_de)],
                    ]
                )
                # then, we consider the contribution of the proximal end
                A_sm_pe = J_pe.T @ jnp.array(
                    [
                        [2 * A_cham * jnp.sin(th_pe), 2 * A_cham * jnp.sin(th_pe)],
                        # [-2 * A_cham * jnp.cos(th_pe), -2 * A_cham * jnp.cos(th_pe)],
                        [-A_cham * r_cop, A_cham * r_cop],
                        [-2 * A_cham * jnp.cos(th_pe), -2 * A_cham * jnp.cos(th_pe)],
                    ]
                )

                # sum the contributions of the distal and proximal ends
                A_sm = A_sm_de + A_sm_pe

            return A_sm

        A_sms = vmap(compute_actuation_matrix_for_segment)(
            self.r_cham_in,
            self.r_cham_out,
            self.varphi_cham,
            chi_pe=chi_tips_list[:-1],
            chi_de=chi_tips_list[1:],
            J_pe=J_tips_list[:-1],
            J_de=J_tips_list[1:],
        )
        # we need to sum the contributions of the actuation of each segment
        A = jnp.sum(A_sms, axis=0)

        # apply the actuation_basis
        A = A @ self.actuation_basis

        return A
