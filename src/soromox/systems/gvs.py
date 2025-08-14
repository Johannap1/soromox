import jax
import jax.numpy as jnp
from jax import vmap, lax, Array

import equinox as eqx

from soromox.utils.basic import (
    compute_strain_basis
)
from soromox.utils.integration import (
    gauss_quadrature,
)
from soromox.utils.lie_algebra import (
    log_SE3,
    exp_gn_SE3, 
    exp_gn_Tg_SE3,
    exp_gn_Tg_Tg_d_SE3,
    adjoint_SE3,
    adjoint_star_SE3,
    Adjoint_g_SE3,
    g_inv_SE3,
)
from soromox.utils.gvs.strain_basis import (
    B_Monomial,
    dof_Monomial,
    B_LegendrePolynomial, 
    dof_LegendrePolynomial,
    B_Chebychev,
    dof_Chebychev,
    B_Fourier,
    dof_Fourier,
    B_Gaussian,
    dof_Gaussian,
    B_IMQ,
    dof_IMQ,    
)
from soromox.utils.gvs.joint_basis import(
    B_Fixed, 
    B_Revolute,
    B_Prismatic,
    B_Helical,
    B_Cylindrical,
    B_Planar,
    B_Spherical, 
    B_Free,
)

# For documentation
from typing import List, Tuple, Callable, Optional
from typing import cast
from soromox.utils.gvs.custom_types import (
    LinkAttributes, 
    JointAttributes, 
    BasisAttributes, 
    SegmentData, 
    JointOperand,
    GeometricOperand,
)

class Basis:    
    BASISTYPE_MAP = {
        'Monomial'  : 0,
        'Legendre'  : 1,
        'Chebychev' : 2,
        'Fourier'   : 3,
        'Gaussian'  : 4,
        'IMQ'       : 5,
    }
    
    DOF_BRANCHES=[
        lambda operand : dof_Monomial(          operand[0], operand[1]),
        lambda operand : dof_LegendrePolynomial(operand[0], operand[1]),
        lambda operand : dof_Chebychev(         operand[0], operand[1]),
        lambda operand : dof_Fourier(           operand[0], operand[1]),
        lambda operand : dof_Gaussian(          operand[0], operand[1]),
        lambda operand : dof_IMQ(               operand[0], operand[1]),
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
        'Revolute'      : 0,    # Revolute joint
        'Prismatic'     : 1,    # Prismatic joint
        'Helical'       : 2,    # Helical joint
        'Cylindrical'   : 3,    # Cylindrical joint
        'Planar'        : 4,    # Planar joint
        'Spherical'     : 5,    # Spherical joint
        'Free'          : 6,    # Free motion joint
        'Fixed'         : 7,    # No motion joint
    }
    AXIS_MAP = {
        'x': 0,  # x-axis
        'y': 1,  # y-axis
        'z': 2,  # z-axis
    }
    PLANE_MAP = {
        'xy': 0,  # xy-plane
        'yz': 1,  # yz-plane
        'xz': 2,  # xz-plane
    }
    DICT_JOINT_TYPE_DOF = {
        'Revolute'      : 1, 
        'Prismatic'     : 1, 
        'Helical'       : 1, 
        'Cylindrical'   : 2, 
        'Planar'        : 3, 
        'Spherical'     : 3, 
        'Free'          : 6,
        'Fixed'         : 0, 
    }
    
    @staticmethod
    def make_B_branch(
        B_fn: Callable[[JointOperand], Array], 
        dof_joint: int, 
        max_dof: int
        )-> Callable:
        def padded_branch(operand):
            B_unpadded = B_fn(operand)

            # Case 1: truncate if dof_joint > max_dof
            if dof_joint > max_dof:
                return B_unpadded[:, :max_dof]

            # Case 2: pad if dof_joint < max_dof
            return jnp.pad(B_unpadded, ((0, 0), (0, max_dof - dof_joint)), constant_values=0.0)

        return padded_branch

class Link:
       
    section_idx             : int
    SECTION_MAP = {
        'Circular'      : 0,  # Circular cross-section
        'Rectangular'   : 1,  # Rectangular cross-section
        'Elliptical'    : 2,  # Elliptical cross-section
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
            Link.compute_elliptical_params
        ]
    
    E              : Array  # Young's Modulus [N/m²]
    nu             : Array  # Poisson Ratio [-1, 0.5]
    G              : Array  # Shear modulus [N/m²]
    rho            : Array  # Density [kg/m³]
    eta            : Array  # Material Damping [N·s/m] 
    l              : Array  # Length of each divisions of the link (soft link) [m]
    
    r   :Tuple[float, float]  # Initial and final value of the geometrical parameter
    h   :Tuple[float, float]  # Initial and final value of the geometrical parameter
    w   :Tuple[float, float]  # Initial and final value of the geometrical parameter
    a   :Tuple[float, float]  # Initial and final value of the geometrical parameter
    b   :Tuple[float, float]  # Initial and final value of the geometrical parameter
    
    @staticmethod
    def interpolate_param(x, a, b):
        return a + (b - a) * x

    @staticmethod
    def compute_rectangular_params(
        operand: GeometricOperand
        )-> Tuple[Array, Array, Array, Array]:
        Xs = operand.Xs
        h_params = operand.h_params
        w_params = operand.w_params
        
        h_nGauss = Link.interpolate_param(Xs, *h_params)
        w_nGauss = Link.interpolate_param(Xs, *w_params)
        
        Iy_p = (1/12) * h_nGauss * w_nGauss**3
        Iz_p = (1/12) * w_nGauss * h_nGauss**3
        Ix_p = Iy_p + Iz_p
        A_p = h_nGauss * w_nGauss
        return Ix_p, Iy_p, Iz_p, A_p
    
    @staticmethod
    def compute_circular_params(
        operand: GeometricOperand
        )-> Tuple[Array, Array, Array, Array]:
        Xs = operand.Xs
        r_params = operand.r_params
        
        r_nGauss = Link.interpolate_param(Xs, *r_params)

        Iy_p = jnp.pi/4 * r_nGauss**4
        Iz_p = Iy_p
        Ix_p = Iy_p + Iz_p
        A_p = jnp.pi * r_nGauss**2
        return Ix_p, Iy_p, Iz_p, A_p
    
    @staticmethod
    def compute_elliptical_params(
        operand: GeometricOperand
        )-> Tuple[Array, Array, Array, Array]:
        Xs = operand.Xs
        a_params = operand.a_params
        b_params = operand.b_params
        
        a_nGauss = Link.interpolate_param(Xs, *a_params)
        b_nGauss = Link.interpolate_param(Xs, *b_params)
        
        Iy_p = jnp.pi/4 * a_nGauss * b_nGauss**3
        Iz_p = jnp.pi/4 * b_nGauss * a_nGauss**3
        Ix_p = Iy_p + Iz_p
        A_p = jnp.pi * a_nGauss * b_nGauss
        return Ix_p, Iy_p, Iz_p, A_p

class System(eqx.Module):
    
    # Static attributes
    N_segment           : int  = eqx.field(static=True) # Number of links in the robot
    max_dof             : int  = eqx.field(static=True) # Maximum number of DOFs for one link of the robot
    max_nGauss          : int  = eqx.field(static=True) # Maximum number of Gauss points for one link of the robot
    max_nip             : int  = eqx.field(static=True) # Maximum number of integration points for one link of the robot
    
    dof_tot_system      : int  = eqx.field(static=True)  # Total number of DOFs for the robot = sum(V_dof)
    dof_tot_max         : int  = eqx.field(static=True)  # Total number of DOFs for the robot, considering the maximum DOFs per link: N_segment * 2 * max_dof
    
    B_select            : Array # Strain basis functions for the robot (6, max_dof)
    
    # Dynamic attributes
    L            : Array  # List of lengths for each link (N_segment, )
    L_cum               : Array  # Cumulative lengths of the links (N_segment + 1, )
    V_nip               : Array   # List of number of evaluation points for each link (N_segment, )
    V_dof               : Array   # List of number of DOFs for each link (N_segment, )
    V_Xs                : Array   # List of integration points for each link (N_segment, max_nip)
    V_Ws                : Array   # List of weights for each link (N_segment, max_nip)
    V_Ms                : Array
    V_Es                : Array
    V_Gs                : Array
    
    V_B_joint           : Array
    V_B_Xs              : Array
    V_B_Z1              : Array
    V_B_Z2              : Array
    
    V_xi_star_joint     : Array
    V_xi_star_Xs        : Array
    V_xi_star_Z1        : Array
    V_xi_star_Z2        : Array
    
    V_K_joint           : Array  # Joint stiffness matrix (N_segment, max_dof, max_dof)
    Gravity_vector      : Array  # Gravity vector (6,)
    
    def __init__(
        self, 
        max_dof     : int,
        max_nGauss  : int,
        links_list  : List[LinkAttributes, ],
        joints_list : List[JointAttributes,],
        basis_list  : List[BasisAttributes,],
        n_gauss_list : List[int], 
        gravity_vector: List[float],
    )-> 'System':
        self.max_dof    = max_dof
        _max_nGauss = max_nGauss
        _max_nip    = _max_nGauss + 2  # +2 for the boundary points        
        self.N_segment  = len(links_list)

        _L     = jnp.empty((self.N_segment, ), dtype=float)
        _V_nip        = jnp.empty((self.N_segment, ), dtype=int)
        _V_dof        = jnp.empty((self.N_segment, 2), dtype=int)
        _V_Xs         = jnp.empty((self.N_segment, _max_nip, ), dtype=float)
        _V_Ws         = jnp.empty((self.N_segment, _max_nip, ), dtype=float)
        
        _V_Ms         = jnp.empty((self.N_segment, _max_nip, 6, 6), dtype=float)
        _V_Es         = jnp.empty((self.N_segment, _max_nip, 6, 6), dtype=float)
        _V_Gs         = jnp.empty((self.N_segment, _max_nip, 6, 6), dtype=float)
        
        _V_B_joint    = jnp.empty((self.N_segment, 6, self.max_dof), dtype=float)
        _V_B_Xs       = jnp.empty((self.N_segment, _max_nip, 6, self.max_dof), dtype=float)
        _V_B_Z1       = jnp.empty((self.N_segment, _max_nip - 1, 6, self.max_dof), dtype=float)
        _V_B_Z2       = jnp.empty((self.N_segment, _max_nip - 1, 6, self.max_dof), dtype=float)
        
        _V_xi_star_joint= jnp.empty((self.N_segment, 6, ), dtype=float)
        _V_xi_star_Xs   = jnp.empty((self.N_segment, _max_nip, 6), dtype=float)
        _V_xi_star_Z1   = jnp.empty((self.N_segment, _max_nip - 1, 6), dtype=float)
        _V_xi_star_Z2   = jnp.empty((self.N_segment, _max_nip - 1, 6), dtype=float)
        
        _V_K_joint = jnp.empty((self.N_segment, self.max_dof, self.max_dof), dtype=float)
        
        V_strain_selector = jnp.zeros((self.N_segment, 2*max_dof), dtype=jnp.bool_)
        
        for i_segment in range(self.N_segment):
            # Use the provided attributes from the links_list            
            joint_attrs    = cast(JointAttributes,  joints_list[i_segment])
            link_attrs     = cast(LinkAttributes,   links_list[i_segment])
            basis_attrs    = cast(BasisAttributes,  basis_list[i_segment])
            n_gauss_i      = n_gauss_list[i_segment]
            
            segment_attributes = self._build_segment(
                max_dof     = self.max_dof,
                max_nip     = _max_nip,
                link_attrs  = link_attrs,
                joint_attrs = joint_attrs,
                basis_attrs = basis_attrs,
                n_gauss     = n_gauss_i,
            )            
            
            # Update the total vectors ==========================================================================           
            _L       = _L.at[i_segment].set(segment_attributes.l)
            _V_nip          = _V_nip.at[i_segment].set(segment_attributes.nip)
            _V_dof          = _V_dof.at[i_segment].set(segment_attributes.dofs_joint_link)
            
            _V_Xs           = _V_Xs.at[i_segment].set(segment_attributes.Xs)
            _V_Ws           = _V_Ws.at[i_segment].set(segment_attributes.Ws)
            
            _V_Ms           = _V_Ms.at[i_segment].set(segment_attributes.Ms)
            _V_Es           = _V_Es.at[i_segment].set(segment_attributes.Es)
            _V_Gs           = _V_Gs.at[i_segment].set(segment_attributes.Gs)
            
            _V_B_joint      = _V_B_joint.at[i_segment].set(segment_attributes.B_joint)
            _V_B_Xs         = _V_B_Xs.at[i_segment].set(segment_attributes.B_Xs)
            _V_B_Z1         = _V_B_Z1.at[i_segment].set(segment_attributes.B_Z1)
            _V_B_Z2         = _V_B_Z2.at[i_segment].set(segment_attributes.B_Z2)
            
            _V_xi_star_joint    = _V_xi_star_joint.at[i_segment].set(segment_attributes.xi_star_joint)
            _V_xi_star_Xs       = _V_xi_star_Xs.at[i_segment].set(segment_attributes.xi_star_Xs)
            _V_xi_star_Z1       = _V_xi_star_Z1.at[i_segment].set(segment_attributes.xi_star_Z1)
            _V_xi_star_Z2       = _V_xi_star_Z2.at[i_segment].set(segment_attributes.xi_star_Z2)
                        
            _V_K_joint = _V_K_joint.at[i_segment].set(segment_attributes.K_joint)
            
            V_strain_selector = V_strain_selector.at[i_segment].set(segment_attributes.strain_selector)
        
        # Strain selector ========================================================
        strain_selector_full = jnp.concatenate(V_strain_selector, axis=0)
        _B_select = compute_strain_basis(strain_selector_full)
        
        _dof_tot_system   = jnp.sum(_V_dof, axis=(0, 1), dtype=int)  # Total DOFs for the robot
        _dof_tot_max   = jnp.array(self.N_segment * 2 * self.max_dof, dtype=int)
        
        # Gravity vector ======================================
        _Gravity_vector = jnp.concatenate([jnp.zeros(3), jnp.array(gravity_vector)])
        
        _L_cum = jnp.cumsum(jnp.concatenate([jnp.zeros(1), _L]))
        
        # Fix attributes
        self.max_nGauss      = _max_nGauss,
        self.max_nip         = _max_nip,
        self.dof_tot_system  = int(_dof_tot_system),
        self.dof_tot_max     = int(_dof_tot_max),
        self.B_select        = _B_select,
        self.L        = _L,
        self.L_cum           = _L_cum,
        self.V_nip           = _V_nip,
        self.V_dof           = _V_dof,
        self.V_Xs            = _V_Xs,
        self.V_Ws            = _V_Ws,
        self.V_Ms            = _V_Ms,
        self.V_Es            = _V_Es,
        self.V_Gs            = _V_Gs,
        self.V_B_joint       = _V_B_joint,
        self.V_B_Xs          = _V_B_Xs,
        self.V_B_Z1          = _V_B_Z1,
        self.V_B_Z2          = _V_B_Z2,
        self.V_xi_star_joint = _V_xi_star_joint,
        self.V_xi_star_Xs    = _V_xi_star_Xs,
        self.V_xi_star_Z1    = _V_xi_star_Z1,
        self.V_xi_star_Z2    = _V_xi_star_Z2,
        self.V_K_joint       = _V_K_joint,
        self.Gravity_vector  = _Gravity_vector,

    def _build_segment(
        self,
        link_attrs: LinkAttributes,
        joint_attrs: JointAttributes,
        basis_attrs: BasisAttributes,
        n_gauss: int,
        max_dof: int,
        max_nip: int
    ) -> SegmentData:
        # Segment ==============================================================================
        # === Joint attributes
        jointtype       = joint_attrs.jointtype
        jointtype_idx   = Joint.JOINTTYPE_MAP[jointtype]
        
        dof_joint = Joint.DICT_JOINT_TYPE_DOF[jointtype]
        K_joint   = jnp.asarray(joint_attrs.K_joint) if jnp.asarray(joint_attrs.K_joint).shape == (dof_joint, dof_joint) else jnp.zeros((dof_joint, dof_joint))
                
        joint_operand = JointOperand(
            axis_idx    =Joint.AXIS_MAP[joint_attrs.axis], 
            plane_idx   =Joint.PLANE_MAP[joint_attrs.plane],
            pitch       =joint_attrs.pitch,
            )
        
        B_joint_full = lax.switch(
            index=jointtype_idx,
            branches=[
                Joint.make_B_branch(B_fn=B_Revolute,    dof_joint=1, max_dof=max_dof),
                Joint.make_B_branch(B_fn=B_Prismatic,   dof_joint=1, max_dof=max_dof),
                Joint.make_B_branch(B_fn=B_Helical,     dof_joint=1, max_dof=max_dof),
                Joint.make_B_branch(B_fn=B_Cylindrical, dof_joint=2, max_dof=max_dof),
                Joint.make_B_branch(B_fn=B_Planar,      dof_joint=3, max_dof=max_dof),
                Joint.make_B_branch(B_fn=B_Spherical,   dof_joint=3, max_dof=max_dof),
                Joint.make_B_branch(B_fn=B_Free,        dof_joint=6, max_dof=max_dof),
                Joint.make_B_branch(B_fn=B_Fixed,       dof_joint=0, max_dof=max_dof),
            ],
            operand=joint_operand
        )
        strain_selector_joint = jnp.concatenate([
            jnp.ones(dof_joint, dtype=bool),
            jnp.zeros(max_dof - dof_joint, dtype=bool)
        ])
        
        xi_star_joint = jnp.zeros((6,))
        K_joint = jnp.asarray(K_joint).reshape((dof_joint, dof_joint))
        K_joint_full = jnp.pad(K_joint, ((0, max_dof - dof_joint), (0, max_dof - dof_joint)), mode='constant') #shape (6, max_dof)
        
        # === Link attributes
        section          = link_attrs.section
        section_idx      = Link.SECTION_MAP[section]
        
        E           = jnp.asarray(link_attrs.E)
        nu          = jnp.asarray(link_attrs.nu)
        rho         = jnp.asarray(link_attrs.rho)
        eta         = jnp.asarray(link_attrs.eta)
        l           = jnp.asarray(link_attrs.l)
        
        r_i         = jnp.asarray(link_attrs.r_i)
        r_f         = jnp.asarray(link_attrs.r_f)
        h_i         = jnp.asarray(link_attrs.h_i)
        h_f         = jnp.asarray(link_attrs.h_f)
        w_i         = jnp.asarray(link_attrs.w_i)
        w_f         = jnp.asarray(link_attrs.w_f)
        a_i         = jnp.asarray(link_attrs.a_i)
        a_f         = jnp.asarray(link_attrs.a_f)
        b_i         = jnp.asarray(link_attrs.b_i)
        b_f         = jnp.asarray(link_attrs.b_f)
        
        G = E / (2 * (1 + nu))  # Shear modulus
        
        r_params = (r_i, r_f)
        h_params = (h_i, h_f)
        w_params = (w_i, w_f)
        a_params = (a_i, a_f)
        b_params = (b_i, b_f)
        
        # === Basis attributes
        basetype        = basis_attrs.basistype
        basistype_idx   = Basis.BASISTYPE_MAP[basetype]
        Bdof    = jnp.asarray(basis_attrs.Bdof).flatten()
        Bodr    = jnp.asarray(basis_attrs.Bodr).flatten()
        xi_star = jnp.asarray(basis_attrs.xi_star).reshape(6,1)
        
        dof_link = lax.switch(
            index=basistype_idx,
            branches=Basis.DOF_BRANCHES,
            operand=(Bdof, Bodr)
        )
        # strain_selector_link = jnp.array(Bdof, dtype=jnp.bool_)
        strain_selector_link = jnp.concatenate([
            jnp.ones(dof_link, dtype=jnp.bool_),
            jnp.zeros(max_dof - dof_link, dtype=jnp.bool_)
        ])
        
        # Zanna points 
        Z1 = 1/2 - jnp.sqrt(3)/6
        Z2 = 1/2 + jnp.sqrt(3)/6
        
        # Gauss points and weights
        Xs, Ws, nip = gauss_quadrature(n_gauss)
        
        deltas = Xs[1:] - Xs[:-1] # Array of shape (nip - 1, )
        
        B_branches = [
            lambda operand: vmap(lambda X:B_Monomial(           X, operand[1], operand[2], max_dof))(operand[0]),
            lambda operand: vmap(lambda X:B_LegendrePolynomial( X, operand[1], operand[2], max_dof))(operand[0]),
            lambda operand: vmap(lambda X:B_Chebychev(          X, operand[1], operand[2], max_dof))(operand[0]),
            lambda operand: vmap(lambda X:B_Fourier(            X, operand[1], operand[2], max_dof))(operand[0]),
            lambda operand: vmap(lambda X:B_Gaussian(           X, operand[1], operand[2], max_dof))(operand[0]),
            lambda operand: vmap(lambda X:B_IMQ(                X, operand[1], operand[2], max_dof))(operand[0]),
        ]
        
        B_Xs  = lax.switch(
            index=basistype_idx,
            branches=B_branches,
            operand=(Xs, Bdof, Bodr)
        )
        B_Z1  = lax.switch(
            index=basistype_idx,
            branches=B_branches,
            operand=(Xs[:-1] + Z1 * deltas, Bdof, Bodr)
        )
        B_Z2  = lax.switch(
            index=basistype_idx,
            branches=B_branches,
            operand=(Xs[:-1] + Z2 * deltas, Bdof, Bodr)
        )
        
        # Compute the initial strain vectors at the integration points
        xi_starfn = lambda x: xi_star #TODO: allow to have an expression for xi_star
        xi_star_Xs = vmap(xi_starfn)(Xs).squeeze()
        xi_star_Z1 = vmap(xi_starfn)(Xs[:-1] + Z1 * deltas).squeeze()
        xi_star_Z2 = vmap(xi_starfn)(Xs[:-1] + Z2 * deltas).squeeze()
        
        # Compute the mass, stiffness, and damping matrices at the integration points        
        geometric_operand = GeometricOperand(
            Xs=Xs,
            r_params=r_params,
            h_params=h_params,
            w_params=w_params,
            a_params=a_params,
            b_params=b_params
        )
        Ix_p, Iy_p, Iz_p, A_p = lax.switch(
            index=section_idx, 
            branches=Link.geometric_branches(),
            operand=geometric_operand
        )
        
        # Préparation des vecteurs composants
        Ms_diag = jnp.stack([Ix_p, Iy_p, Iz_p, A_p, A_p, A_p], axis=1)  # Shape: (np, 6)
        Es_diag = jnp.stack([G * Ix_p, E * Iy_p, E * Iz_p, E * A_p, G * A_p, G * A_p], axis=1)
        Gs_diag = jnp.stack([Ix_p, 3 * Iy_p, 3 * Iz_p, 3 * A_p, A_p, A_p], axis=1)
        
        Ms = rho * vmap(jnp.diag)(Ms_diag)  # Shape: (np, 6, 6)
        Es = vmap(jnp.diag)(Es_diag)
        Gs = eta * vmap(jnp.diag)(Gs_diag)

        # Pad the arrays to the maximum number of integration points and DOFs 
        Xs_full = jnp.pad(Xs, (0, max_nip - nip), mode='constant')
        Ws_full = jnp.pad(Ws, (0, max_nip - nip), mode='constant')
        
        Ms_full = jnp.pad(Ms, ((0, max_nip - nip), (0, 0), (0, 0)), mode='constant')
        Es_full = jnp.pad(Es, ((0, max_nip - nip), (0, 0), (0, 0)), mode='constant')
        Gs_full = jnp.pad(Gs, ((0, max_nip - nip), (0, 0), (0, 0)), mode='constant')
        
        B_Xs_full = jnp.pad(B_Xs, ((0, max_nip - nip), (0, 0), (0, 0)), mode='constant')
        B_Z1_full = jnp.pad(B_Z1, ((0, max_nip - nip), (0, 0), (0, 0)), mode='constant')
        B_Z2_full = jnp.pad(B_Z2, ((0, max_nip - nip), (0, 0), (0, 0)), mode='constant')
        
        xi_star_Xs_full = jnp.pad(xi_star_Xs, ((0, max_nip - nip), (0, 0)), mode='constant')
        xi_star_Z1_full = jnp.pad(xi_star_Z1, ((0, max_nip - nip), (0, 0)), mode='constant')
        xi_star_Z2_full = jnp.pad(xi_star_Z2, ((0, max_nip - nip), (0, 0)), mode='constant')

        dofs_joint_link = jnp.stack([dof_joint, dof_link])
        
        strain_selector_segment = jnp.concatenate([
                strain_selector_joint, 
                strain_selector_link
            ], axis=0)
        
        return SegmentData(
            l               = l,
            nip             = nip,
            dofs_joint_link = dofs_joint_link,
            strain_selector = strain_selector_segment,
            Xs              = Xs_full,
            Ws              = Ws_full,
            Ms              = Ms_full,
            Es              = Es_full,
            Gs              = Gs_full,
            B_joint         = B_joint_full,
            B_Xs            = B_Xs_full,
            B_Z1            = B_Z1_full,
            B_Z2            = B_Z2_full,
            xi_star_joint   = xi_star_joint,
            xi_star_Xs      = xi_star_Xs_full,
            xi_star_Z1      = xi_star_Z1_full,
            xi_star_Z2      = xi_star_Z2_full,
            K_joint         = K_joint_full
        )
    
    # Gathering functions =========================================================
    def _min_size_gathered(self, vec_min_size_flat: Array)-> Array:
        """
        Function to split the joint coordinates into segments and pad them to the maximum DOF.
        
        Args:
            vec_min_size_flat (Array): shape (dof_tot, 1) or (dof_tot,) JAX array
                Joint coordinates or joint velocities.
        Returns:
            vec_gathered (Array): shape (N_segment, 2, max_dof) JAX array
                Padded joint coordinates.
        """
        vec = jnp.asarray(vec_min_size_flat).reshape(-1)  # Ensure q is a 1D array
        
        _V_dof = self.V_dof  # Shape (N_segment, 2)
        
        m, n_groups = _V_dof.shape  # n_groups = 2

        # Start index calculation
        V_dof_flat = _V_dof.reshape(-1)
        start_indices_flat = jnp.cumsum(jnp.pad(V_dof_flat, (1, 0)))[:-1]
        start_indices = start_indices_flat.reshape(m, n_groups)

        # Function for obtaining indexes
        def get_indices(start, length):
            idx = jnp.arange(self.max_dof)
            return start + idx

        # Vectorization: apply to (m, 2)
        get_indices_vmap = vmap(vmap(get_indices, in_axes=(0, 0)), in_axes=(0, 0))
        all_indices = get_indices_vmap(start_indices, _V_dof)

        # Creating masks
        def get_mask(length):
            return jnp.arange(self.max_dof) < length

        get_mask_vmap = vmap(vmap(get_mask, in_axes=(0,)), in_axes=(0,))
        mask = get_mask_vmap(_V_dof)

        # Secure padding
        vec_padded = jnp.concatenate([vec, jnp.zeros(self.max_dof)])

        # We retrieve the values
        vec_gathered = jnp.take(vec_padded, all_indices, mode='clip') * mask

        return vec_gathered

    def _max_size_gathered(self, vec_max_size: Array) -> Array:
        """
        Gather the configuration vector into the shape (N_segment, 2, max_dof).
        
        Args:
            vec_max_size (Array): configuration vector of shape (N_segment * 2 * max_dof, )
        Returns:
            vec_gathered (Array): gathered configuration vector of shape (N_segment, 2, max_dof)
        """
        vec_gathered = vec_max_size.reshape(self.N_segment, 2, self.max_dof)
        return vec_gathered
    
    # =========================================================================
    def _forward_kinematics(self, q_gathered: Array)-> Array:
        """
        Function to compute the forward kinematics of the linkage.

        Args:
            q_gathered (Array): shape (N_segment, 2, max_dof) JAX array
                Joint coordinates.

        Returns:
            g_list (Array): shape (N_segment, max_nip, 4, 4) JAX array
                Forward kinematics transformation matrices at all significant points
        """
        _max_nip = self.max_nip
        
        # Joint properties =========================        
        _V_B_joint = self.V_B_joint  # shape (N_segment, 6, max_dof)
        _V_xi_star_joint = self.V_xi_star_joint  # shape (N_segment, 6)
        
        # Link properties =========================
        _V_Xs           = self.V_Xs         # shape (N_segment, max_nip)
        _V_xi_star_Z1   = self.V_xi_star_Z1 # shape (N_segment, max_nip - 1, 6)  
        _V_xi_star_Z2   = self.V_xi_star_Z2 # shape (N_segment, max_nip - 1, 6)
        _L       = self.L     # shape (N_segment, 1,)
        _V_B_Z1         = self.V_B_Z1       # shape (N_segment, max_nip - 1, 6, max_dof)
        _V_B_Z2         = self.V_B_Z2       # shape (N_segment, max_nip - 1, 6, max_dof)
        
        def body_segment_i(carry, i_segment):
            g_tip = carry
            
            # Joint =======================
            B_joint_i       = _V_B_joint[i_segment]        # shape (6, max_dof)
            xi_star_joint_i = _V_xi_star_joint[i_segment]  # shape (6,)
            q_joint_i       = q_gathered[i_segment, 0]     # shape (max_dof,)
            
            xi_joint_i  = B_joint_i @ q_joint_i + xi_star_joint_i  # shape (6,)
            
            g_joint_i   = exp_gn_SE3(xi_joint_i)  # shape (4, 4)
            
            g_j = g_tip @ g_joint_i  # Start with the last transformation matrix
                        
            # Link ========================
            Xs_i            = _V_Xs[i_segment]         # shape (max_nip,)
            xi_star_Z1_i    = _V_xi_star_Z1[i_segment] # shape (max_nip - 1, 6)  
            xi_star_Z2_i    = _V_xi_star_Z2[i_segment] # shape (max_nip - 1, 6)
            length_i        = _L[i_segment]     # shape (1,)
            B_Z1_i          = _V_B_Z1[i_segment]       # shape (max_nip - 1, 6, max_dof)
            B_Z2_i          = _V_B_Z2[i_segment]       # shape (max_nip - 1, 6, max_dof)
            
            q_i = q_gathered[i_segment, 1]
            
            def body_eval_points(carry, j_eval):
                g_j_prev = carry
                                
                H = Xs_i[j_eval + 1] - Xs_i[j_eval]
                
                xi_star_Z1_j = xi_star_Z1_i[j_eval].at[:3].multiply(length_i)
                xi_star_Z2_j = xi_star_Z2_i[j_eval].at[:3].multiply(length_i)
                
                B_Z1_j = B_Z1_i[j_eval]
                B_Z2_j = B_Z2_i[j_eval]
                
                xi_Z1_j = B_Z1_j @ q_i + xi_star_Z1_j
                xi_Z2_j = B_Z2_j @ q_i + xi_star_Z2_j
                
                # Magnus expansion
                ad_xi_Z1_j  = adjoint_SE3(xi_Z1_j)
                Magnus_j    = (
                    (H / 2) * (xi_Z1_j + xi_Z2_j)
                    + (jnp.sqrt(3) * H ** 2 / 12) * (ad_xi_Z1_j @ xi_Z2_j)
                )

                Magnus_j    = Magnus_j.at[3:6].multiply(length_i)
                g_step      = exp_gn_SE3(Magnus_j)

                g_j = g_j_prev @ g_step
                                
                return g_j, g_j
            
            indices_eval_points = jnp.arange(_max_nip - 1)
            
            g_tip_link, g_link = lax.scan(
                f=body_eval_points,
                init=g_j,  # Start with the last transformation matrix
                xs=indices_eval_points
            )
            
            # # For debugging purposes, you can uncomment the following lines to see the step-by-step computation
            # carry = g_j
            # g_link = []
            # for x in indices_eval_points:
            #     g_j, g_j = body_eval_points(carry, x)
            #     carry = g_j
            #     g_link.append(g_j)
            # g_tip_link = g_j
            # g_link = jnp.array(g_link)
            
            g_link = jnp.concatenate((jnp.expand_dims(g_j, axis=0), g_link), axis=0)
                        
            return g_tip_link, g_link
    
        indices_link = jnp.arange(0, self.N_segment)
        
        g_ini = jnp.eye(4)  # Initial transformation matrix (identity)
        
        _, g_list = lax.scan(
            f=body_segment_i,
            init=g_ini,
            xs=indices_link
        )
        
        # # For debugging purposes, you can uncomment the following lines to see the step-by-step computation
        # carry = g_ini
        # g_list = []
        # for x in indices_link:
        #     g_tip_link, g_link = body_segment_i(carry, x)
        #     carry = g_tip_link
        #     g_list.append(g_link)
        # g_tip_final = g_tip_link
        # g_list = jnp.array(g_list)
        
        return g_list
    
    def classify_segment(
        self,
        s: Array,
    ) -> Tuple[Array, Array]:
        """
        Classify the point along the robot to the corresponding segment.

        Args:
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            segment_idx (Array): index of the segment where the point is located
            s_segment (Array): point coordinate along the segment in the interval [0, l_segment]
        """

        # Classify the point along the robot to the corresponding segment
        segment_idx = jnp.clip(jnp.sum(s > self.L_cum) - 1, 0, self.N_segment - 1)

        # Compute the point coordinate along the segment in the interval [0, l_segment]
        s_local = s - self.L_cum[segment_idx]

        return segment_idx, s_local
    
    def get_pose_at_s(self, q_gathered, s):
        g_list = self._forward_kinematics(q_gathered)  # shape (N_segment, max_nip, 4, 4)
        lengths = self.L.flatten()
        
        segment_idx, s_local = self.classify_segment(s)  # segment_idx: index of the segment, s_local: point coordinate along the segment
        length_local = lengths[segment_idx]
        X = s_local / length_local

        Xs = self.V_Xs[segment_idx]
        idx_interp = jnp.searchsorted(Xs, X, side="right") - 1
        t = (X - Xs[idx_interp]) / (Xs[idx_interp + 1] - Xs[idx_interp])

        g1 = g_list[segment_idx, idx_interp]
        g2 = g_list[segment_idx, idx_interp + 1]

        g1_inv = jnp.linalg.inv(g1)
        delta_g = g1_inv @ g2
        xi = log_SE3(delta_g)
        g_interp = g1 @ exp_gn_SE3(t * xi)

        return g_interp
    
    def jacobian(
        self,
        q_gathered: Array,
        )-> Array:
        """
        Function to compute the Jacobian of the linkage.

        Args:
            q_gathered (Array): shape (N_segment, 2, max_dof) JAX array
                Joint coordinates.

        Returns:
            J_list (Array): shape (N_segment, max_nip, N_segment, 2, 6, max_dof) JAX array
                Jacobian matrices at all significant points
                J_list[i_segment, j_nip, k_link, type_contrib, s, d]
                    i_segment : link for which we calculate the Jacobian
                    j_nip : evaluation point index on i_segment
                    k_link : contributing link (i.e. the part of the kinematic chain that influences this point)
                    type_contrib ∈ {0,1} : type of contribution :
                        0 = articulation (joint)
                        1 = rigid segment (Magnus step)
                    s ∈ {0..5} : spatial components of viscinematics (twist)
                    d ∈ {0..max_dof-1} : partial derivative with respect to the d-th DoF of the link k_link
        """
        _max_nip    = self.max_nip
        
        # Joint properties =========================        
        _V_B_joint       = self.V_B_joint  # shape (N_segment, 6, max_dof)
        _V_xi_star_joint = self.V_xi_star_joint  # shape (N_segment, 6)
        
        # Link properties =========================
        _V_Xs           = self.V_Xs         # shape (N_segment, max_nip)
        _V_xi_star_Z1   = self.V_xi_star_Z1 # shape (N_segment, max_nip - 1, 6)  
        _V_xi_star_Z2   = self.V_xi_star_Z2 # shape (N_segment, max_nip - 1, 6)
        _L       = self.L     # shape (N_segment, 1,)
        _V_B_Z1         = self.V_B_Z1       # shape (N_segment, max_nip - 1, 6, max_dof)
        _V_B_Z2         = self.V_B_Z2       # shape (N_segment, max_nip - 1, 6, max_dof)
        
        def body_segment_i(carry, i_segment):
            g_tip, J_tip = carry

            # Joint ============================
            B_joint_i       = _V_B_joint[i_segment]  # (6, max_dof)
            xi_star_joint_i = _V_xi_star_joint[i_segment]  # (6,)
            q_joint_i       = q_gathered[i_segment, 0]

            xi_joint_i  = B_joint_i @ q_joint_i + xi_star_joint_i
            
            g_joint_i, T_g_joint = exp_gn_Tg_SE3(xi_joint_i)
            
            product_T_g_joint_B_joint_i     = T_g_joint @ B_joint_i  # shape (N_segment, 6, max_dof)
            T_g_joint_i_B_joint_i           = jnp.zeros((self.N_segment, 2, 6, self.max_dof)).at[i_segment, 0].set(
                        product_T_g_joint_B_joint_i
                    )
            
            Ad_g_joint_inv = Adjoint_g_SE3(g_inv_SE3(g_joint_i))        
            
            g_j_scaled = g_tip @ g_joint_i
            J_j_scaled = jnp.einsum('ij,nmjk->nmik', Ad_g_joint_inv, (J_tip + T_g_joint_i_B_joint_i))
            
            # Link ========================
            Xs_i            = _V_Xs[i_segment]         # shape (max_nip,)
            xi_star_Z1_i    = _V_xi_star_Z1[i_segment] # shape (max_nip - 1, 6)  
            xi_star_Z2_i    = _V_xi_star_Z2[i_segment] # shape (max_nip - 1, 6)
            length_i        = _L[i_segment]     # shape (1,)
            B_Z1_i          = _V_B_Z1[i_segment]       # shape (max_nip - 1, 6, max_dof)
            B_Z2_i          = _V_B_Z2[i_segment]       # shape (max_nip - 1, 6, max_dof)
            
            g_j = g_j_scaled.at[0:3, 3].divide(length_i)  # shape (4, 4)
            J_j = J_j_scaled.at[:, :, 3:6, :].divide(length_i)  # shape (N_segment, 6, max_dof)
            
            q_i = q_gathered[i_segment, 1]
            
            def body_eval_points(carry, j_eval):
                g_j, J_j = carry

                H = Xs_i[j_eval + 1] - Xs_i[j_eval]

                xi_star_Z1_j = xi_star_Z1_i[j_eval].at[:3].multiply(length_i)
                xi_star_Z2_j = xi_star_Z2_i[j_eval].at[:3].multiply(length_i)

                B_Z1_j = B_Z1_i[j_eval]
                B_Z2_j = B_Z2_i[j_eval]

                xi_Z1_j = B_Z1_j @ q_i + xi_star_Z1_j
                xi_Z2_j = B_Z2_j @ q_i + xi_star_Z2_j

                ad_xi_Z1_j = adjoint_SE3(xi_Z1_j)
                ad_xi_Z2_j = adjoint_SE3(xi_Z2_j)

                Magnus_j = (
                    (H / 2) * (xi_Z1_j + xi_Z2_j) 
                    + (jnp.sqrt(3) * H ** 2 / 12) * (ad_xi_Z1_j @ xi_Z2_j)
                )

                B_Magnus_j = (
                    (H / 2) * (B_Z1_j + B_Z2_j) 
                    + (jnp.sqrt(3) * H ** 2 / 12) 
                        * (ad_xi_Z1_j @ B_Z2_j - ad_xi_Z2_j @ B_Z1_j)
                )
                
                g_step, T_step = exp_gn_Tg_SE3(Magnus_j)
                
                product_T_step_B_Magnus_j = T_step @ B_Magnus_j
                
                T_step_B_step = jnp.zeros((self.N_segment, 2, 6, self.max_dof)).at[i_segment, 1].set(
                    product_T_step_B_Magnus_j
                )
                
                Ad_g_step_inv = Adjoint_g_SE3(g_inv_SE3(g_step))
                
                g_j = g_j @ g_step
                J_j = jnp.einsum('ij,nmjk->nmik', Ad_g_step_inv, (J_j + T_step_B_step)) # shape (N_segment, 6, max_dof)

                J_j_scaled = J_j.at[:, :, 3:6, :].multiply(length_i)
                                
                return (g_j, J_j), J_j_scaled

            indices_eval_points = jnp.arange(_max_nip - 1)

            (g_tip_link, J_tip_link), J_link = lax.scan(
                f=body_eval_points,
                init=(g_j, J_j),
                xs=indices_eval_points
            )
            
            # # For debugging purposes, you can uncomment the following lines to see the step-by-step computation
            # carry = (g_j, J_j)
            # J_link = []
            # for x in indices_eval_points:
            #     (g_j, J_j), J_j_scaled_here = body_eval_points(carry, x)
            #     carry = (g_j, J_j)
            #     J_link.append(J_j_scaled_here)
            # (g_tip_link, J_tip_link) = carry
            # J_link = jnp.array(J_link)  # shape (max_nip - 1, N_segment, 6, max_dof)
            
            J_link = jnp.concatenate((jnp.expand_dims(J_j_scaled, axis=0), J_link), axis=0)
            
            g_tip_link_scaled = g_tip_link.at[0:3, 3].multiply(length_i)
            J_tip_link_scaled = J_tip_link.at[:, :, 3:6, :].multiply(length_i)

            return (g_tip_link_scaled, J_tip_link_scaled), J_link

        indices_link = jnp.arange(0, self.N_segment)

        g_init = jnp.eye(4)
        J_init = jnp.zeros((self.N_segment, 2, 6, self.max_dof))
        
        _, J_list = lax.scan(
            f=body_segment_i,
            init=(g_init, J_init),
            xs=indices_link
        )
        
        # # For debugging purposes, you can uncomment the following lines to see the step-by-step computation
        # carry = (g_init, J_init)
        # J_list = []
        # for x in indices_link:
        #     (g_tip_link, J_tip_link), J_j = body_segment_i(carry, x)
        #     carry = (g_tip_link, J_tip_link)
        #     J_list.append(J_j)
        # J_list = jnp.array(J_list)
        
        # (N_segment, _max_nip, N_segment, 2, 6, max_dof) => (N_segment, max_nip, 6, N_segment*2*max_dof)
        J_reordered = jnp.transpose(J_list, (0, 1, 4, 2, 3, 5))
        J = J_reordered.reshape(self.N_segment, _max_nip, 6, self.N_segment * 2 * self.max_dof)
                
        return J
    
    def jacobian_d(
        self,
        q_gathered: Array,
        q_d_gathered: Array,
        )-> Array:
        """
        Function to compute the Jacobian derivative of the linkage.

        Args:
            q_gathered (Array): shape (N_segment, 2, max_dof) JAX array
                Joint coordinates.
                
            q_d_gathered (Array): shape (N_segment, 2, max_dof) JAX array
                Joint velocities.

        Returns:
            J_d_list (Array): shape (N_segment, max_nip, 6, N_segment*2*max_dof) JAX array
                Jacobian derivative matrices at all significant points
        """
        _max_nip    = self.max_nip
         
        # Joint properties =========================        
        _V_B_joint = self.V_B_joint  # shape (N_segment, 6, max_dof)
        _V_xi_star_joint = self.V_xi_star_joint  # shape (N_segment, 6)
        
        # Link properties =========================
        _V_Xs           = self.V_Xs         # shape (N_segment, max_nip)
        _V_xi_star_Z1   = self.V_xi_star_Z1 # shape (N_segment, max_nip - 1, 6)  
        _V_xi_star_Z2   = self.V_xi_star_Z2 # shape (N_segment, max_nip - 1, 6)
        _L       = self.L     # shape (N_segment, 1,)
        _V_B_Z1         = self.V_B_Z1       # shape (N_segment, max_nip - 1, 6, max_dof)
        _V_B_Z2         = self.V_B_Z2       # shape (N_segment, max_nip - 1, 6, max_dof)

        def body_segment_i(carry, i_segment):
            g_tip, J_d_tip, eta_tip = carry

            # Joint ============================
            B_joint_i       = _V_B_joint[i_segment]        # shape (6, max_dof)
            xi_star_joint_i = _V_xi_star_joint[i_segment]  # shape (6,)
            q_joint_i       = q_gathered[i_segment, 0]     # shape (max_dof,)
            q_d_joint_i   = q_d_gathered[i_segment, 0] # shape (max_dof,)

            xi_joint_i      = B_joint_i @ q_joint_i + xi_star_joint_i   # shape (6,)
            xi_d_joint_i  = B_joint_i @ q_d_joint_i                 # shape (6,)
            
            g_joint_i, T_g_joint, T_d_g_joint = exp_gn_Tg_Tg_d_SE3(xi_joint_i, xi_d_joint_i)

            product_T_g_joint_B_joint_i         = T_g_joint @ B_joint_i  # shape (6, max_dof)
            product_T_d_g_joint_B_joint_i     = T_d_g_joint @ B_joint_i  # shape (6, max_dof)

            T_g_joint_B_joint_i = product_T_g_joint_B_joint_i # shape (6, max_dof)
            T_d_g_joint_B_joint_i = jnp.zeros((self.N_segment, 2, 6, self.max_dof)).at[i_segment, 0].set(
                adjoint_SE3(eta_tip) @ (product_T_g_joint_B_joint_i) 
                + product_T_d_g_joint_B_joint_i
            ) # shape (N_segment, 2, 6, max_dof)
            
            Ad_g_joint_inv = Adjoint_g_SE3(g_inv_SE3(g_joint_i))
            
            g_j_scaled       = g_tip @ g_joint_i
            J_d_j_scaled   = jnp.einsum('ij,nmjk->nmik', Ad_g_joint_inv, J_d_tip + T_d_g_joint_B_joint_i)
            eta_j_scaled     = Ad_g_joint_inv@(eta_tip + T_g_joint_B_joint_i @ q_d_joint_i)
            
            # Link ========================
            Xs_i            = _V_Xs[i_segment]         # shape (max_nip,)
            xi_star_Z1_i    = _V_xi_star_Z1[i_segment] # shape (max_nip - 1, 6)  
            xi_star_Z2_i    = _V_xi_star_Z2[i_segment] # shape (max_nip - 1, 6)
            length_i        = _L[i_segment]     # shape (1,)
            B_Z1_i          = _V_B_Z1[i_segment]       # shape (max_nip - 1, 6, max_dof)
            B_Z2_i          = _V_B_Z2[i_segment]       # shape (max_nip - 1, 6, max_dof)
            
            q_i     = q_gathered[i_segment, 1]
            q_d_i   = q_d_gathered[i_segment, 1]
            
            g_j         = g_j_scaled.at[0:3, 3].divide(length_i)  
            J_d_j     = J_d_j_scaled.at[:, :, 3:6, :].divide(length_i)
            eta_j       = eta_j_scaled.at[3:6].divide(length_i)

            def body_eval_points(carry, j_eval):
                g_j, J_d_j, eta_j = carry

                H = Xs_i[j_eval + 1] - Xs_i[j_eval]

                xi_star_Z1_j = xi_star_Z1_i[j_eval].at[:3].multiply(length_i)
                xi_star_Z2_j = xi_star_Z2_i[j_eval].at[:3].multiply(length_i)

                B_Z1_j = B_Z1_i[j_eval]
                B_Z2_j = B_Z2_i[j_eval]

                xi_Z1_j     = B_Z1_j @ q_i + xi_star_Z1_j
                xi_Z2_j     = B_Z2_j @ q_i + xi_star_Z2_j
                xi_d_Z1_j = B_Z1_j @ q_d_i

                ad_xi_Z1_j = adjoint_SE3(xi_Z1_j)
                ad_xi_Z2_j = adjoint_SE3(xi_Z2_j)

                Magnus_j = (
                    (H / 2) * (xi_Z1_j + xi_Z2_j) 
                    + (jnp.sqrt(3) * H ** 2 / 12) * (ad_xi_Z1_j @ xi_Z2_j)
                )

                B_Magnus_j = (
                    (H / 2) * (B_Z1_j + B_Z2_j) 
                    + (jnp.sqrt(3) * H ** 2 / 12) 
                        * (ad_xi_Z1_j @ B_Z2_j - ad_xi_Z2_j @ B_Z1_j)
                )

                Magnus_d_j = B_Magnus_j @ q_d_i
                
                Magnus_dd_dq_j = ((jnp.sqrt(3) * H ** 2) / 6) * adjoint_SE3(xi_d_Z1_j) @ B_Z2_j
                
                g_step, T_Magnus_j, T_d_Magnus_j = exp_gn_Tg_Tg_d_SE3(Magnus_j, Magnus_d_j)

                product_T_Magnus_j_B_Magnus_j = T_Magnus_j @ B_Magnus_j
                product_T_d_Magnus_j_B_Magnus_j = T_d_Magnus_j @ B_Magnus_j
                product_T_Magnus_j_Magnus_dd_dq_j = T_Magnus_j @ Magnus_dd_dq_j
                
                T_B_Magnus_step = product_T_Magnus_j_B_Magnus_j # shape (6, max_dof)
                T_B_Magnus_d_step = jnp.zeros((self.N_segment, 2, 6, self.max_dof)).at[i_segment, 1].set(
                    adjoint_SE3(eta_j) @ product_T_Magnus_j_B_Magnus_j 
                    + product_T_d_Magnus_j_B_Magnus_j
                    + product_T_Magnus_j_Magnus_dd_dq_j
                ) #shape (N_segment, 2, 6, max_dof)
                
                Ad_g_step_inv = Adjoint_g_SE3(g_inv_SE3(g_step)) # shape (N_segment, 6, 6)
                g_j         = g_j @ g_step
                J_d_j     = jnp.einsum('ij,nmjk->nmik', Ad_g_step_inv, J_d_j + T_B_Magnus_d_step)
                eta_j       = Ad_g_step_inv@(eta_j + T_B_Magnus_step @ q_d_i)
                
                J_d_j_scaled = J_d_j.at[:, :, 3:6, :].multiply(length_i)

                return (g_j, J_d_j, eta_j), J_d_j_scaled

            indices_eval_points = jnp.arange(_max_nip - 1)
            
            (g_tip_link, J_d_tip_link, eta_tip_link), J_d_link = lax.scan(
                f=body_eval_points,
                init=(g_j, J_d_j, eta_j),
                xs=indices_eval_points
            )

            # For debugging purposes, you can uncomment the following lines to see the step-by-step computation
            # carry = (g_j, J_d_j, eta_j)
            # J_d_link = []
            # for x in indices_eval_points:
            #     (g_j, J_d_j, eta_j), J_d_j_scaled_here = body_eval_points(carry, x)
            #     carry = (g_j, J_d_j, eta_j)
            #     J_d_link.append(J_d_j_scaled_here)
            # (g_tip_link, J_d_tip_link, eta_tip_link) = carry
            # J_d_link = jnp.array(J_d_link)

            J_d_link = jnp.concatenate((jnp.expand_dims(J_d_j_scaled, axis=0), J_d_link), axis=0)
            
            g_tip_link_scaled = g_tip_link.at[0:3, 3].multiply(length_i)
            J_d_tip_link_scaled = J_d_tip_link.at[:, :, 3:6, :].multiply(length_i)
            eta_tip_link_scaled = eta_tip_link.at[3:6].multiply(length_i)
            
            return (g_tip_link_scaled, J_d_tip_link_scaled, eta_tip_link_scaled), J_d_link

        indices_link = jnp.arange(0, self.N_segment)

        g_init = jnp.eye(4)
        J_d_init = jnp.zeros((self.N_segment, 2, 6, self.max_dof))
        eta_init = jnp.zeros((6,))
        
        _, J_d_list = lax.scan(
            f=body_segment_i,
            init=(g_init, J_d_init, eta_init),
            xs=indices_link
        )

        # For debugging purposes, you can uncomment the following lines to see the step-by-step computation
        # carry = (g_init, J_d_init, eta_init)
        # J_d_list = []
        # for x in indices_link:
        #     (g_tip_link, J_d_tip_link, eta_tip_link), J_d_j = body_segment_i(carry, x)
        #     carry = (g_tip_link, J_d_tip_link, eta_tip_link)
        #     J_d_list.append(J_d_j)
        # J_d_list = jnp.array(J_d_list)
    
        # (N_segment, _max_nip, N_segment, 2, 6, max_dof) => (N_segment, max_nip, 6, N_segment*2*max_dof)
        J_d_reordered = jnp.transpose(J_d_list, (0, 1, 4, 2, 3, 5))        
        J_d = J_d_reordered.reshape(self.N_segment, _max_nip, 6, self.N_segment * 2 * self.max_dof)

        return J_d
    
    def inertia_matrix(
        self, 
        q_gathered: Array
        ) -> Array:
        """
        Compute the inertia matrix B(q) using vectorized operations with vmap.

        Args:
            q_gathered (Array): shape (N_segment, 2, max_dof) JAX array
                Joint coordinates.

        Returns:
            B (Array): Inertia matrix, shape (N_segment * 2 * max_dof, N_segment * 2 * max_dof)
        """
        _max_nip    = self.max_nip
        
        _V_Ms       = self.V_Ms         # (N_segment, max_nip, 6, 6)
        _V_Ws       = self.V_Ws         # (N_segment, max_nip, 6, max_dof)
        _L   = self.L     # (N_segment, 1)
        
        V_J = self.jacobian(q_gathered)           # (N_segment, max_nip, 6, N_segment * 2 * max_dof)

        # Define function for each quadrature point
        def B_segment_i(i_segment):
            length_i    = _L[i_segment]
            Ws_i        = _V_Ws[i_segment]     # (max_nip, 1, )
            J_i         = V_J[i_segment]       # (max_nip, 6, N_segment * 2 * max_dof)
            Ms_i        = _V_Ms[i_segment]     # (max_nip, 6, 6)
        
            def B_eval_points(i_eval): 
                Ws_j    = Ws_i[i_eval]
                J_j     = J_i[i_eval]           # (6, N_segment * 2 * max_dof)
                Ms_j    = Ms_i[i_eval]          # (6, 6)
                
                return Ws_j * (J_j.T @ Ms_j @ J_j) # (N_segment * 2 * max_dof, N_segment * 2 * max_dof)
            
            B_blocks_segment_i = vmap(B_eval_points)(jnp.arange(_max_nip)) # (max_nip, N_segment * 2 * max_dof, N_segment * 2 * max_dof)

            # # For debugging purposes, we can use a list comprehension
            # B_blocks_segment_i = jnp.stack([B_eval_points(i_eval) for i_eval in range(_max_nip)])  # (max_nip, N_segment * 2 * max_dof, N_segment * 2 * max_dof)
            
            return B_blocks_segment_i * length_i
        
        B_blocks_tot = vmap(B_segment_i)(jnp.arange(self.N_segment)) # (N_segment, max_nip, N_segment * 2 * max_dof, N_segment * 2 * max_dof)
        
        # # For debugging purposes, we can use a list comprehension
        # B_blocks_tot = jnp.stack([B_segment_i(i_segment) for i_segment in range(self.N_segment)])  # (N_segment, max_nip, N_segment * 2 * max_dof, N_segment * 2 * max_dof)
        
        B = jnp.sum(B_blocks_tot, axis=(0, 1))

        return B
    
    def coriolis_matrix(
        self, 
        q_gathered: Array, 
        q_d_gathered: Array
        ) -> Array:
        """
        Function to compute the generalized Coriolis matrix of the linkage.

        Args:
            q_gathered (Array): shape (N_segment, 2, max_dof) JAX array
                Joint coordinates.
            q_d_gathered (Array): shape (N_segment, 2, max_dof) JAX array
                Joint velocities.

        Returns:
            C (Array): Coriolis matrix, shape (N_segment * 2 * max_dof, N_segment * 2 * max_dof)
        """
        _max_nip    = self.max_nip
        
        _V_Ms       = self.V_Ms         # (N_segment, max_nip, 6, 6)
        _V_Ws       = self.V_Ws         # (N_segment, max_nip, 6, max_dof)
        _L   = self.L     # (N_segment, 1)
        
        V_J       = self.jacobian(q_gathered)            # (N_segment, max_nip, 6, N_segment * 2 * max_dof)
        V_J_d     = self.jacobian_d(q_gathered, q_d_gathered) # (N_segment, max_nip, 6, N_segment * 2 * max_dof)
        
        q_d_flat = q_d_gathered.reshape(-1)
        
        # Define function for each quadrature point
        def C_segment_i(i_segment):
            length_i    = _L[i_segment]
            Ws_i        = _V_Ws[i_segment]      # (max_nip, 1, )
            J_i         = V_J[i_segment]        # (max_nip, 6, N_segment * 2 * max_dof)
            J_d_i       = V_J_d[i_segment]      # (max_nip, 6, N_segment * 2 * max_dof)
            Ms_i        = _V_Ms[i_segment]      # (max_nip, 6, 6)
        
            def C_eval_points(i_eval): 
                Ws_j    = Ws_i[i_eval]
                J_j     = J_i[i_eval]    # (6, N_segment * 2 * max_dof)
                J_d_j   = J_d_i[i_eval]  # (6, N_segment * 2 * max_dof)
                Ms_j    = Ms_i[i_eval]   # (6, 6)
                
                return Ws_j *(J_j.T @ (Ms_j @ J_d_j + adjoint_star_SE3(J_j @ q_d_flat) @ Ms_j @ J_j)) # (N_segment * 2 * max_dof, N_segment * 2 * max_dof)
            
            C_blocks_segment_i = vmap(C_eval_points)(jnp.arange(_max_nip)) # (max_nip, N_segment * 2 * max_dof, N_segment * 2 * max_dof)

            # For debugging purposes, we can use a list comprehension
            # C_blocks_segment_i = jnp.stack([C_eval_points(i_eval) for i_eval in range(_max_nip)])  # (max_nip, N_segment * 2 * max_dof, N_segment * 2 * max_dof)
            
            return C_blocks_segment_i * length_i
        
        C_blocks_tot = vmap(C_segment_i)(jnp.arange(self.N_segment)) # (N_segment, max_nip, N_segment * 2 * max_dof, N_segment * 2 * max_dof)
        
        # For debugging purposes, we can use a list comprehension
        # C_blocks_tot = jnp.stack([C_segment_i(i_segment) for i_segment in range(self.N_segment)])  # (N_segment, max_nip, N_segment * 2 * max_dof, N_segment * 2 * max_dof)
        
        C = jnp.sum(C_blocks_tot, axis=(0, 1))

        return C
    
    def gravitational_matrix(
        self,
        q_gathered: Array
        ) -> Array:
        """
        Function to compute the generalized external force of the linkage.
        
        Args:
            q_gathered (Array): shape (N_segment, 2, max_dof) JAX array
                Joint coordinates.
        
        Returns:
            G (Array): Gravitational matrix, shape (N_segment * 2 * max_dof, 1)
        """
        _max_nip    = self.max_nip

        _Gravity    = self.Gravity_vector  # (6, 1) gravitational acceleration vector
        _V_Ms       = self.V_Ms     # (N_segment, max_nip, 6, 6)
        _V_Ws       = self.V_Ws     # (N_segment, max_nip, 6, max_dof)
        _L   = self.L # (N_segment, 1)
        
        V_J = self.jacobian(q_gathered)              # (N_segment, max_nip, 6, N_segment * 2 * max_dof)
        V_g = self._forward_kinematics(q_gathered)    # (N_segment, max_nip, 4, 4)
        
        def G_segment_i(i_segment):
            length_i    = _L[i_segment]
            Ws_i        = _V_Ws[i_segment]  # (max_nip, 1, )
            g_i         = V_g[i_segment]    # (max_nip, 4, 4)
            J_i         = V_J[i_segment]    # (max_nip, 6, N_segment * 2 * max_dof)
            M_i         = _V_Ms[i_segment]  # (max_nip, 6, 6)
            
            def G_eval_points(i_eval):
                Ws_j    = Ws_i[i_eval]          # ()
                g_j     = g_i[i_eval]           # (4, 4)
                Ad_g_j_inv = Adjoint_g_SE3(g_inv_SE3(g_j)) # (6, 6)
                J_j     = J_i[i_eval]           # (6, N_segment * 2 * max_dof)
                M_j     = M_i[i_eval]           # (6, 6)
                
                return Ws_j * J_j.T @ M_j @ Ad_g_j_inv @ _Gravity # (N_segment * 2 * max_dof, 1)
            
            G_blocks_segment_i = vmap(G_eval_points)(jnp.arange(_max_nip)) # (max_nip, N_segment * 2 * max_dof, 1)

            # For debugging purposes, we can use a list comprehension
            # G_blocks_segment_i = jnp.stack([G_eval_points(i_eval) for i_eval in range(_max_nip)])
            
            return G_blocks_segment_i * length_i
            
        G_blocks_tot = vmap(G_segment_i)(jnp.arange(self.N_segment)) # (N_segment, max_nip, N_segment * 2 * max_dof, 1)
        
        # For debugging purposes, we can use a list comprehension
        # G_blocks_tot = jnp.stack([G_segment_i(i_segment) for i_segment in range(self.N_segment)])  # (N_segment, max_nip, N_segment * 2 * max_dof, 1)
        
        G = jnp.sum(G_blocks_tot, axis=(0, 1))  # Sum over links and quadrature points
        
        return G

    def stiffness_matrix(
        self
        ) -> Array:
        """
        Function to compute the generalized stiffness matrix for the linkage.

        Returns:
            K (Array): Global stiffness matrix of shape (N_segment * 2 * max_dof, N_segment * 2 * max_dof)
        """
        _max_nip   = self.max_nip
        
        _V_K_joint = self.V_K_joint  # (N_segment, max_dof, max_dof)
        
        _V_Es      = self.V_Es      # (N_segment, max_nip, 6, 6)
        _V_Ws      = self.V_Ws      # (N_segment, max_nip, 6, max_dof)
        _L  = self.L  # (N_segment, 1)
        _V_B_Xs    = self.V_B_Xs    # (N_segment, max_nip, 6, max_dof)

        def K_segment_i(i_segment):
            # Joint ==============================
            K_joint_i = jnp.zeros((self.max_dof, self.max_dof))#_V_K_joint[i_segment]  # (max_dof, max_dof) TODO
            
            # Link ===============================
            length_i    = _L[i_segment]
            Ws_i        = _V_Ws[i_segment]     # (max_nip, 1, )
            Es_i        = _V_Es[i_segment]     # (max_nip, 6, 6)
            B_Xs_i      = _V_B_Xs[i_segment]   # (max_nip, 6, max_dof)

            def K_eval_points(i_eval):
                Ws_j    = Ws_i[i_eval]
                Es_j    = Es_i[i_eval]           # (6, 6)
                B_Xs_j  = B_Xs_i[i_eval]         # (6, max_dof)
                
                Es_j_scaled = Es_j.at[:3, :].divide(length_i**3).at[3:, :].divide(length_i)
                
                return Ws_j * (B_Xs_j.T @ Es_j_scaled @ B_Xs_j)

            K_link_i = jnp.sum(vmap(K_eval_points)(jnp.arange(_max_nip)), axis=0) * length_i**2  # (max_nip, max_dof, max_dof)
            
            # Create a (2, max_dof, max_dof) array with K_joint_i and K_segment_i
            K_blocks_segment_i = jnp.stack([K_joint_i, K_link_i], axis=0)
            return K_blocks_segment_i
        
        K_blocks_tot = vmap(K_segment_i)(jnp.arange(self.N_segment))  # (N_segment, 2, max_dof, max_dof)
                
        # Assume that K_blocks is of the form (N_segment, 2, max_dof, max_dof)
        K_blocks_flat = K_blocks_tot.reshape(-1, self.max_dof, self.max_dof)  # (N_segment * 2, max_dof, max_dof)

        # Convert to list of matrices
        K_blocks_list = [K_blocks_flat[i] for i in range(K_blocks_flat.shape[0])]

        # Building the diagonal matrix in blocks
        K = jax.scipy.linalg.block_diag(*K_blocks_list)
                
        return K
    
    def damping_matrix(
        self
        ) -> Array:
        """
        Function to compute the generalized damping matrix for the linkage.

        Returns:
            K (Array): Global stiffness matrix of shape (N_segment * 2 * max_dof, N_segment * 2 * max_dof)
        """
        _max_nip   = self.max_nip
        
        # _V_B_joint = self.V_B_joint # (N_segment, 6, max_dof)
        
        _V_Gs      = self.V_Gs      # (N_segment, max_nip, 6, 6)
        _V_Ws      = self.V_Ws      # (N_segment, max_nip, 6, max_dof)
        _L  = self.L  # (N_segment, 1)
        _V_B_Xs    = self.V_B_Xs    # (N_segment, max_nip, 6, max_dof)

        def D_segment_i(i_segment):
            # Joint ============================== TODO
            D_joint_i = jnp.zeros((self.max_dof, self.max_dof))  # Initialize joint stiffness matrix
            
            # Link ===============================
            length_i    = _L[i_segment]
            Ws_i        = _V_Ws[i_segment]     # (max_nip, 1, )
            Gs_i        = _V_Gs[i_segment]     # (max_nip, 6, 6)
            B_Xs_i      = _V_B_Xs[i_segment]   # (max_nip, 6, max_dof)

            def D_eval_points(i_eval):
                Ws_j    = Ws_i[i_eval]
                Gs_j    = Gs_i[i_eval]           # (6, 6)
                B_Xs_j  = B_Xs_i[i_eval]         # (6, max_dof)
                
                Gs_j_scaled = Gs_j.at[:3, :].divide(length_i**3).at[3:, :].divide(length_i)
                
                return Ws_j * (B_Xs_j.T @ Gs_j_scaled @ B_Xs_j)

            D_link_i = jnp.sum(vmap(D_eval_points)(jnp.arange(_max_nip)), axis=0) * length_i**2  # (max_nip, max_dof, max_dof)
            
            # Create a (2, max_dof, max_dof) array with D_joint_i and D_segment_i
            D_blocks_segment_i = jnp.stack([D_joint_i, D_link_i], axis=0)
            return D_blocks_segment_i
        
        D_blocks_tot = vmap(D_segment_i)(jnp.arange(self.N_segment))  # (N_segment, 2, max_dof, max_dof)
                
        # Supposons que D_blocks est de forme (N_segment, 2, max_dof, max_dof)
        D_blocks_flat = D_blocks_tot.reshape(-1, self.max_dof, self.max_dof)  # (N_segment * 2, max_dof, max_dof)

        # Convertir en liste de matrices
        D_blocks_list = [D_blocks_flat[i] for i in range(D_blocks_flat.shape[0])]

        # Construire la matrice diagonale par blocs
        D = jax.scipy.linalg.block_diag(*D_blocks_list)
                
        return D
    
    def dynamical_matrices(
        self,
        q_min_size: Array,
        q_d_min_size: Array
        ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """
        Compute the dynamical matrices of the system.
        
        Args:
            q_min_size: configuration vector of shape (dof_tot, )
            q_d_min_size: configuration velocity vector of shape (dof_tot, )
        
        Returns:
            B: Inertia matrix of shape (dof_tot, dof_tot)
            C: Coriolis matrix of shape (dof_tot, dof_tot)
            G: Gravitational matrix of shape (N_segment * 2 * max_dof, 1)
            K: Stiffness matrix of shape (dof_tot, dof_tot)
            D: Damping matrix of shape (dof_tot, dof_tot)
            A: Identity matrix of shape (dof_tot, dof_tot)
        """
        q_gathered = self._min_size_gathered(q_min_size)         # shape (N_segment*2*max_dof)
        q_d_gathered = self._min_size_gathered(q_d_min_size)     # shape (N_segment*2*max_dof)
        
        B_full = self.inertia_matrix(q_gathered)
        C_full = self.coriolis_matrix(q_gathered, q_d_gathered)
        G_full = self.gravitational_matrix(q_gathered)
        K_full = self.stiffness_matrix()
        D_full = self.damping_matrix()
        
        _B_select = self.B_select
        # _B_select = compute_strain_basis(self.strain_selector)
        
        # Extract the relevant parts of the matrices
        B = _B_select.T @ B_full @ _B_select    # Inertia matrix
        C = _B_select.T @ C_full @ _B_select    # Coriolis matrix
        G = _B_select.T @ G_full            # Gravitational matrix
        K = _B_select.T @ K_full @ _B_select    # Stiffness matrix
        D = _B_select.T @ D_full @ _B_select    # Damping matrix
        A = jnp.eye(self.dof_tot_system)# Identity matrix TODO
        return B, C, G, K, D, A
    
    def dynamical_matrices_max(
        self,
        q_max_size: Array,
        q_d_max_size: Array
        ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """
        Compute the dynamical matrices of the system.
        
        Args:
            q_max_size: configuration vector of shape (dof_tot_max, )
            q_d_max_size: configuration velocity vector of shape (dof_tot_max, )
        
        Returns:
            B: Inertia matrix of shape (dof_tot_max, dof_tot_max)
            C: Coriolis matrix of shape (dof_tot_max, dof_tot_max)
            G: Gravitational matrix of shape (dof_tot_max, 1)
            K: Stiffness matrix of shape (dof_tot_max, dof_tot_max)
            D: Damping matrix of shape (dof_tot_max, dof_tot_max)
            A: Identity matrix of shape (dof_tot_max, dof_tot_max)
        """
        q_gathered      = self._max_size_gathered(q_max_size)         # shape (N_segment*2*max_dof)
        q_d_gathered    = self._max_size_gathered(q_d_max_size)     # shape (N_segment*2*max_dof)
        
        B = self.inertia_matrix(q_gathered)
        C = self.coriolis_matrix(q_gathered, q_d_gathered)
        G = self.gravitational_matrix(q_gathered)
        K = self.stiffness_matrix()
        D = self.damping_matrix()
        A = jnp.eye(self.dof_tot_max)   # Identity matrix TODO
        
        return B, C, G, K, D, A
    
    def forward_dynamics(
        self,
        q_min_size: Array,
        q_d_min_size: Array,
        tau_min_size: Optional[Array] = None
    ) -> Array:
        """
        Compute the forward dynamics of the system.

        Args:
            q_min_size (Array): Joint coordinates of shape (dof_tot, )
            q_d_min_size (Array): Joint velocities of shape (dof_tot, )
            tau_min_size (Optional[Array]): Joint torques of shape (dof_tot, ). If None, it defaults to zero.

        Returns:
            q_dd_min_size (Array): Joint accelerations of shape (dof_tot, )
        """
        B, C, G, K, D, A = self.dynamical_matrices(q_min_size, q_d_min_size)
        
        if tau_min_size is None:
            tau_min_size = jnp.zeros((self.dof_tot_system,))
        
        RHS = A @ tau_min_size - C @ q_d_min_size - G - K @ q_min_size - D @ q_d_min_size
        
        # Solve for joint accelerations
        q_dd_min_size = jnp.linalg.solve(B, RHS)
        
        return q_dd_min_size
    
    def forward_dynamics_max(
        self,
        q_min_size: Array,
        q_d_min_size: Array,
        tau_min_size: Optional[Array] = None
    ) -> Array:
        """
        Compute the forward dynamics of the system.

        Args:
            q_min_size (Array): Joint coordinates of shape (dof_tot, )
            q_d_min_size (Array): Joint velocities of shape (dof_tot, )
            tau_min_size (Optional[Array]): Joint torques of shape (dof_tot, ). If None, it defaults to zero.
            
        Returns:
            q_dd_max_size (Array): Joint accelerations of shape (dof_tot_max, )
        """        
        q_gathered      = self._min_size_gathered(q_min_size)         # shape (N_segment*2*max_dof)
        q_d_gathered    = self._min_size_gathered(q_d_min_size)     # shape (N_segment*2*max_dof)
        
        q_max_size      = q_gathered.reshape(-1)  # Flatten the array for processing
        q_d_max_size    = q_d_gathered.reshape(-1)  # Flatten the array for processing
        
        B = self.inertia_matrix(q_gathered)
        C = self.coriolis_matrix(q_gathered, q_d_gathered)
        G = self.gravitational_matrix(q_gathered)
        K = self.stiffness_matrix()
        D = self.damping_matrix()
        A = jnp.eye(self.dof_tot_max)   # Identity matrix TODO
        
        if tau_min_size is None:
            tau_min_size = jnp.zeros((self.dof_tot_system,))
        tau_max_size = self._min_size_gathered(tau_min_size).reshape(-1)
        
        RHS = A @ tau_max_size - C @ q_d_max_size - G - K @ q_max_size - D @ q_d_max_size
        
        # Solve for joint accelerations using the Tichonov regularization method
        q_dd_max_size = jnp.linalg.inv(B + jnp.finfo(jnp.float64).eps*jnp.eye(self.dof_tot_max)) @ RHS
        
        return q_dd_max_size
