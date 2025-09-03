__all__ = ["LinkData", "SegmentData"]
from dataclasses import dataclass, field
import jax
from jax import Array


@jax.tree_util.register_pytree_node_class
@dataclass
class SegmentData:
    L: Array  # Length of the segment
    nip: Array  # Number of integration points
    dofs_joint_link: Array  # Degrees of freedom of the segment as [dof_joint, dof_link]
    strain_selector: (
        Array  # Boolean array indicating which strain components are active
    )
    Xs: Array  # Integration points
    Ws: Array  # Weights for the integration points
    Ms: Array  # Mass matrices at integration points
    Es: Array  # Stiffness matrices at integration points
    Gs: Array  # Damping matrices at integration points
    B_joint: Array  # Joint basis matrix
    B_Xs: Array  # Basis matrix at integration points
    B_Z1: Array  # Basis matrix at Z1 points
    B_Z2: Array  # Basis matrix at Z2 points
    xi_ref_joint: Array  # Joint initial strain vector
    xi_ref_Xs: Array  # Initial strain vector at integration points
    xi_ref_Z1: Array  # Initial strain vector at Z1 points
    xi_ref_Z2: Array  # Initial strain vector at Z2 points
    K_joint: Array  # Joint stiffness matrix

    def tree_flatten(self):
        children = (
            self.L,
            self.nip,
            self.dofs_joint_link,
            self.Xs,
            self.Ws,
            self.Ms,
            self.Es,
            self.Gs,
            self.B_joint,
            self.B_Xs,
            self.B_Z1,
            self.B_Z2,
            self.xi_ref_joint,
            self.xi_ref_Xs,
            self.xi_ref_Z1,
            self.xi_ref_Z2,
            self.K_joint,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass
class LinkData:
    L: Array  # Length of the segment
    nip: Array  # Number of integration points
    strain_selector: (
        Array  # Boolean array indicating which strain components are active
    )
    Xs: Array  # Integration points
    Ws: Array  # Weights for the integration points
    Ms: Array  # Mass matrices at integration points
    Es: Array  # Stiffness matrices at integration points
    Gs: Array  # Damping matrices at integration points
    B_Xs: Array  # Basis matrix at integration points
    B_Z1: Array  # Basis matrix at Z1 points
    B_Z2: Array  # Basis matrix at Z2 points
    xi_ref_Xs: Array  # Initial strain vector at integration points
    xi_ref_Z1: Array  # Initial strain vector at Z1 points
    xi_ref_Z2: Array  # Initial strain vector at Z2 points
    dof_link: Array  # Degrees of freedom of the segment

    def tree_flatten(self):
        children = (
            self.L,
            self.nip,
            self.Xs,
            self.Ws,
            self.Ms,
            self.Es,
            self.Gs,
            self.B_Xs,
            self.B_Z1,
            self.B_Z2,
            self.xi_ref_Xs,
            self.xi_ref_Z1,
            self.xi_ref_Z2,
            self.dof_link,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
