import jax
import jax.numpy as jnp
from soromox.systems.gvs.attributes import LinkAttributes, JointAttributes, BasisAttributes
from soromox.systems.gvs.core import GVS
import optimistix as optx
from jax import lax
import numpy as _np


def solve_equilibrium(gvs, u, q0):
    def statics_eq(q,args): 
        gvs, u = args
        K = gvs.stiffness_matrix()
        B = gvs.actuation_matrix(q)
        F = gvs.gravitational_force(q)
        return K @ q - F - B @ u
    

    solver = optx.Newton(rtol=1e-6, atol=1e-6)
    res = optx.root_find(jax.jit(statics_eq), solver, q0, args=(gvs, u), max_steps=50)

    # print("Soluzione trovata q*:", res.value)
    # print("Residuo finale:", jnp.linalg.norm(statics_eq(res.value, (gvs, u))))
    return res





# def main():
# Link 1
link1 = LinkAttributes(
    section="Circular",
    E=3.02e5,
    nu=0.4,
    rho=1310.0,
    eta=1e4,
    L=0.025,
    r_i=0.01541,
    r_f=0.01467,
)
# Link 2
link2 = LinkAttributes(
    section="Circular",
    E=3.02e5,
    nu=0.4,
    rho=1310.0,
    eta=1e4,
    L=0.2550,
    r_i=0.01467,
    r_f=0.00716,
)
  #Link 3
link3 = LinkAttributes(
    section="Circular",
    E=3.02e5,
    nu=0.4,
    rho=1310.0,
    eta=1e4,
    L=0.0250,
    r_i=0.00716,
    r_f=0.00642,
)
# Link 4
link4 = LinkAttributes(
    section="Circular",
    E=3.02e5,
    nu=0.4,
    rho=1310.0,
    eta=1e4,
    L=0.0550,
    r_i=0.00642,
    r_f=0.00480,
)
joint1 = JointAttributes(jointtype="Fixed")
joint2 = JointAttributes(jointtype="Fixed")
joint3 = JointAttributes(jointtype="Fixed")
joint4 = JointAttributes(jointtype="Fixed")
basis1 = BasisAttributes(basistype="Monomial", Bdof=[0, 1, 1, 1, 0, 0], Bodr=[0, 0, 0, 0, 0, 0])
basis2 = BasisAttributes(basistype="Monomial", Bdof=[1, 1, 1, 1, 0, 0], Bodr=[1, 1, 1, 1, 0, 0])
basis3 = BasisAttributes(basistype="Monomial", Bdof=[1, 1, 1, 1, 0, 0], Bodr=[0, 0, 0, 0, 0, 0])
basis4 = BasisAttributes(basistype="Monomial", Bdof=[0, 1, 1, 0, 0, 0], Bodr=[0, 0, 0, 0, 0, 0])
n_gauss_list = [5, 8, 5, 5]
# gravity_vector = [9.81, 0.0, 0.0]
gravity_vector = [0.0, 0.0, -9.81]

gvs = GVS(
    links_list=[link1, link2, link3, link4],
    joints_list=[joint1, joint2, joint3, joint4],
    basis_list=[basis1, basis2, basis3, basis4],
    n_gauss_list=n_gauss_list,
    gravity_vector=gravity_vector,
)
print("model built")
print("num_segments:", int(gvs.num_segments))
print("num_dofs:", int(gvs.num_actuators))
print("B_select shape:", gvs.B_select.shape)
print("B_select:", gvs.B_select)
print("V_dof (joint, link) per segment:\n", gvs.V_dof)
print("max_dof: \n",gvs.max_dof)
max_dof = gvs.max_dof
padded = 2 * max_dof

xi_reference = jax.device_get(gvs.V_xi_ref_Xs)
print("xi_reference shape:", _np.array(xi_reference).shape)
print(_np.array(xi_reference))
xi_ref_b1gp0 = xi_reference[0, 0]
print("xi_ref segment 0, gauss 0:", _np.array(xi_ref_b1gp0))


for j in range(gvs.num_segments):
    dof_joint = int(gvs.V_dof[j, 0])
    dof_link = int(gvs.V_dof[j, 1])
    
    n_gauss_seg = int(gvs.V_nip[j])

    print(f"segment {j} -> dof_joint={dof_joint}, dof_link={dof_link}")

    B = jax.device_get(gvs.B_select) 
    start_seg = j * padded
    start_joint = start_seg
    start_link = start_seg + max_dof
    n_active = B.shape[1]
    padded_link_block = B[start_link : start_link + max_dof, :n_active]
    compact_link_block = padded_link_block[:dof_link, :]

    print("B_select compact (segment 0, link) shape:", compact_link_block.shape)
    print(compact_link_block)


    #bx0 ha dim data (6, max_dof)
    BX0 = jax.device_get(gvs.V_B_Xs[j, 0])  # usual shape (6, max_dof)
    print("  BX0 shape:", _np.array(BX0).shape)
    print(_np.array(BX0))
    Bq_link = BX0[:, :dof_link]
    print("  Bq_link shape:", _np.array(Bq_link).shape)
    print(_np.array(Bq_link))

    #bx_all ha dim data (num_gauss_max, 6, max_dof)
    BX_all = jax.device_get(gvs.V_B_Xs[j])
    print("  BX_all shape:", _np.array(BX_all).shape)
    print(_np.array(BX_all))

    BX_used = BX_all[:n_gauss_seg, :, :dof_link]    # shape (n_gauss_seg, 6, dof_link)
    print("  BX_used shape (gauss, 6, dof_link):", _np.array(BX_used).shape)

    V_flat = gvs.V_dof.reshape(-1)
    starts = jnp.concatenate([jnp.array([0], dtype=V_flat.dtype), jnp.cumsum(V_flat)[:-1]])
    start_col_link = starts[2 * j + 1]
    B_xi_i = jnp.zeros((6, gvs.B_select.shape[1]))
    BXs = jax.device_get(gvs.V_B_Xs[j, 1])
    B_xi_i = lax.dynamic_update_slice(B_xi_i, BXs[:, :gvs.V_dof[j, 1]], (0, start_col_link))

    print("  B_xi_i shape (6, n_active):", _np.array(B_xi_i).shape)
    print(_np.array(B_xi_i))

# dof = gvs.num_actuators

# q0 = jnp.zeros((dof,))
# q0dot = jnp.zeros((dof,))
# L_cum = jax.device_get(gvs.V_L_cum)
# total_length = float(L_cum[-1])
# s_end = float(total_length)
# g_end = gvs.forward_kinematics(q0, s_end)

# J_end = gvs.jacobian_bodyframe(q0, s_end)
# M = gvs.inertia_matrix(q0)
# K = gvs.stiffness_matrix()
# F = gvs.gravitational_force(q0)
# D = gvs.damping_matrix()
# B = gvs.actuation_matrix(q0)
# u = jnp.zeros((dof,))

# #THE INITIAL CONDITION CAN BE CHANGED WITH G_INI IN CORE (914 G,1298 J, 1754 JDOT)

# #PRIMa DI INTERVENIRE SULLA MATRICE DI ATTUAZIONE, PROVA A STAMPARE QUA B_SELECT E I SINGOLI DOF
# #COSI CAPISCI COME AGIRE

# print("L_cum:", L_cum)  
# print("total length:", total_length)
# print("s_end:", s_end)
# print("g(s_end) SE(3):\n", g_end)
# print("Jacobian at end (6 x dof):\n", J_end)
# print("Inertia matrix:", M)
# print("Stiffness matrix:", K)
# print("Gravitational force:", F)
# print("Damping matrix:", D)
# print("Actuation matrix:", B )
  
# y = jnp.concatenate([q0, q0dot])
# ydot = gvs.forward_dynamics(0.0, y)
# print("ydot (q0dot, q0ddot):", ydot)

# res = solve_equilibrium(gvs, u, q0)
# print("q* =", res.value)

