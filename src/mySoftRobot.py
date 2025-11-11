import jax
import pandas as pd
import jax.numpy as jnp
from soromox.systems.gvs.attributes import LinkAttributes, JointAttributes, BasisAttributes
#from soromox.systems.gvs.core import GVS
from soromox.systems.gvs.tendon_actuated_gvs import TendonActuatedGVS
from functools import partial
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import optimistix as optx
import optax
from jax import Array, lax
import numpy as onp

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "gpu")  # or "cpu"

print("JAX default backend:", jax.default_backend())
print("JAX devices:", jax.devices())

### MOCAP DATA TRANSFORMATION FUNCTIONS ###

def _transform_points(pb: jnp.ndarray, p1: jnp.ndarray, p2: jnp.ndarray, p3: jnp.ndarray, p4: jnp.ndarray,
                      Rot: jnp.ndarray, R_alg1: jnp.ndarray, R_alg2: jnp.ndarray, offset: jnp.ndarray):
    # equivalente del blocco richiesto
    pb_gl = Rot @ pb
    p1_gl = Rot @ p1
    p2_gl = Rot @ p2
    p3_gl = Rot @ p3
    p4_gl = Rot @ p4

    pb_lcl = -offset
    p1_lcl = p1_gl - (pb_gl + offset)
    p2_lcl = p2_gl - (pb_gl + offset)
    p3_lcl = p3_gl - (pb_gl + offset)
    p4_lcl = p4_gl - (pb_gl + offset)

    R_total = R_alg1 @ R_alg2
    pb_tr = R_total @ pb_lcl
    p1_tr = R_total @ p1_lcl
    p2_tr = R_total @ p2_lcl
    p3_tr = R_total @ p3_lcl
    p4_tr = R_total @ p4_lcl

    return {"pb_tr": pb_tr, "p1_tr": p1_tr, "p2_tr": p2_tr, "p3_tr": p3_tr, "p4_tr": p4_tr}

### SOFT ROBOT UTILITIES FUNCTIONS ###

# MY STATIC EQUILIBRIUM EQUATION SOLVER
def solve_equilibrium(robot, u, q0):
    def statics_eq(q,args): 
        robot, u = args
        K = robot.stiffness_matrix()
        B = robot.actuation_matrix(q)
        G = robot.gravitational_force(q)
        return K @ q - G - B @ u
    

    solver = optx.Newton(rtol=1e-6, atol=1e-6)
    res = optx.root_find(jax.jit(statics_eq), solver, q0, args=(robot, u), max_steps=50)

    # print("Solution q*:", res.value)
    # print("Final residual:", jnp.linalg.norm(statics_eq(res.value, (robot, u))))
    return res


# DRAWING FUNCTIONS 
jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)

def draw_robot_curve(
    robot: TendonActuatedGVS,
    q: Array,
    num_points: int = 50,
):
    batched_forward_kinematics = jax.vmap(robot.forward_kinematics, in_axes=(None, 0))
    L_max = jnp.sum(robot.V_L)

    s_ps = jnp.linspace(0, L_max, num_points)
    g_ps = batched_forward_kinematics(q, s_ps)[:, :3, 3]

    curve = onp.array(g_ps, dtype=onp.float64)
    return curve  # (N, 3)

def animate_robot_matplotlib(
    robot: TendonActuatedGVS,
    t_list: Array,  # shape (T,)
    q_list: Array,  # shape (T, DOF)
    interval: int = 50,
    slider: bool = None,
    animation: bool = None,
    show: bool = True,
):
    if slider is None and animation is None:
        raise ValueError("Either 'slider' or 'animation' must be set to True.")
    if animation and slider:
        raise ValueError(
            "Cannot use both animation and slider at the same time. Choose one."
        )

    width = jnp.linalg.norm(robot.V_L) * 3
    height = width

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])  # [left, bottom, width, height]

    if animation:
        (line,) = ax.plot([], [], [], lw=4, color="blue")
        ax.set_xlim(-width / 2, width / 2)
        ax.set_ylim(-width / 2, width / 2)
        ax.set_zlim(0, height)
        title_text = ax.set_title("t = 0.00 s")

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            title_text.set_text("t = 0.00 s")
            return line, title_text

        def update(frame_idx):
            q = q_list[frame_idx]
            t = t_list[frame_idx]
            curve = draw_robot_curve(robot, q)
            line.set_data(curve[:, 0], curve[:, 1])
            line.set_3d_properties(curve[:, 2])
            title_text.set_text(f"t = {t:.2f} s")
            return line, title_text

        ani = FuncAnimation(
            fig,
            update,
            frames=len(q_list),
            init_func=init,
            blit=False,
            interval=interval,
        )

        if show:
            plt.show()

        plt.close(fig)
        return HTML(ani.to_jshtml())

    elif slider:

        def update_plot(frame_idx):
            ax.cla()  # Clear current axes
            ax.set_xlim(-width / 2, width / 2)
            ax.set_ylim(-width / 2, width / 2)
            ax.set_zlim(0, height)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_zlabel("Z [m]")
            ax.set_title(f"t = {t_list[frame_idx]:.2f} s")
            q = q_list[frame_idx]
            curve = draw_robot_curve(robot, q)
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], lw=4, color="blue")
            fig.canvas.draw_idle()

        # Create slider
        slider = Slider(
            ax=ax_slider,
            label="Frame",
            valmin=0,
            valmax=len(t_list) - 1,
            valinit=0,
            valstep=1,
        )
        slider.on_changed(update_plot)

        update_plot(0)  # Initial plot

        if show:
            plt.show()

        plt.close(fig)
        return HTML(
            "Slider animation not implemented in HTML format. Use matplotlib directly to view the slider."
        )  # Slider cannot be converted to HTML



### BODY DEFINITION OF THE SOFT ROBOT ###

#2 link version
# Link 1
link1 = LinkAttributes(
    section="Circular",
    E=3.04e5,
    nu=0.45,
    rho=1310.0,
    eta=1e4,
    L=0.0250+0.2550+0.0250,
    r_i=0.01541,
    r_f=0.00642,
)

# Link 2
link2 = LinkAttributes(
    section="Circular",
    E=3.04e5,
    nu=0.45,
    rho=1310.0,
    eta=1e4,
    L=0.0550,
    r_i=0.00642,
    r_f=0.00480,
)
joint1 = JointAttributes(jointtype="Fixed")
joint2 = JointAttributes(jointtype="Fixed")

basis1 = BasisAttributes(basistype="Monomial", Bdof=[1, 1, 1, 1, 0, 0], Bodr=[1, 1, 1, 1, 0, 0])
basis2 = BasisAttributes(basistype="Monomial", Bdof=[0, 1, 1, 0, 0, 0], Bodr=[0, 0, 0, 0, 0, 0])


n_gauss_list = [8, 8]
gravity_vector = [0.0, 0.0, -9.81]


tendon_routing_params = {
     "ry": jnp.array([0.0114*jnp.cos(jnp.pi/180*30), 0.0114*jnp.cos(jnp.pi/180*150)]),
     "my": jnp.array([-0.0295*jnp.cos(jnp.pi/180*30),-0.0295*jnp.cos(jnp.pi/180*150)]),
     "rz": jnp.array([0.0114*jnp.sin(jnp.pi/180*30), 0.0114*jnp.sin(jnp.pi/180*150)]),
     "mz": jnp.array([-0.0295*jnp.sin(jnp.pi/180*30), -0.0295*jnp.sin(jnp.pi/180*150)]),
     "idx_seg_att": jnp.array([0, 0]),
}
#REMEMBER: 0DEG -> Y+, 180DEG -> Y-, 90DEG -> Z+, 270DEG -> Z-

# 2 link version
robot = TendonActuatedGVS(
    links_list=[link1, link2],
    joints_list=[joint1, joint2],
    basis_list=[basis1, basis2],
    n_gauss_list=n_gauss_list,
    gravity_vector=gravity_vector,
    tendon_routing_params=tendon_routing_params,
)


### MATRICES CHECKING AND PRINTING ###

print("num_segments:", int(robot.num_segments))
print("V_dof (joint, link) per segment:\n", robot.V_dof)
print("max_dof: \n",robot.max_dof)
# max_dof = robot.max_dof
# padded = 2 * max_dof


#### DEBUG: CHECKING MATRICES #####

# xi_reference = jax.device_get(robot.V_xi_ref_Xs)
# print("xi_reference shape:", onp.array(xi_reference).shape)
# print(onp.array(xi_reference))
# xi_ref_b1gp0 = xi_reference[0, 0]
# print("xi_ref segment 0, gauss 0:", onp.array(xi_ref_b1gp0))

#Bq printing per segment
for j in range(robot.num_segments):
    dof_joint = int(robot.V_dof[j, 0])
    dof_link = int(robot.V_dof[j, 1])
    
#     n_gauss_seg = int(robot.V_nip[j])

#     print(f"segment {j} -> dof_joint={dof_joint}, dof_link={dof_link}")

#     B = jax.device_get(robot.B_select) 
#     start_seg = j * padded
#     start_joint = start_seg
#     start_link = start_seg + max_dof
#     n_active = B.shape[1]
#     padded_link_block = B[start_link : start_link + max_dof, :n_active]
#     compact_link_block = padded_link_block[:dof_link, :]

#     print("B_select compact (segment 0, link) shape:", compact_link_block.shape)
#     print(compact_link_block)

    # print("robot.B_select.shape[1]", robot.B_select.shape[1])
    # #bx0 ha dim data (6, max_dof)
    # BX0 = jax.device_get(robot.V_B_Xs[j, 0])  # usual shape (6, max_dof)
    # print("  BX0 shape:", onp.array(BX0).shape)
    # print(onp.array(BX0))
    # Bq_link = BX0[:, :dof_link]
    # print("  Bq_link shape:", onp.array(Bq_link).shape)
    # print(onp.array(Bq_link))

#     #bx_all ha dim data (num_gauss_max, 6, max_dof)
#     BX_all = jax.device_get(robot.V_B_Xs[j])
#     print("  BX_all shape:", onp.array(BX_all).shape)
#     print(onp.array(BX_all))

#     BX_used = BX_all[:n_gauss_seg, :, :dof_link]    # shape (n_gauss_seg, 6, dof_link)
#     print("  BX_used shape (gauss, 6, dof_link):", onp.array(BX_used).shape)

#     V_flat = robot.V_dof.reshape(-1)
#     starts = jnp.concatenate([jnp.array([0], dtype=V_flat.dtype), jnp.cumsum(V_flat)[:-1]])
#     start_col_link = starts[2 * j + 1]
#     B_xi_i = jnp.zeros((6, robot.B_select.shape[1]))
#     BXs = jax.device_get(robot.V_B_Xs[j, 1])
#     B_xi_i = lax.dynamic_update_slice(B_xi_i, BXs[:, :robot.V_dof[j, 1]], (0, start_col_link))

#     print("  B_xi_i shape (6, n_active):", onp.array(B_xi_i).shape)
#     print(onp.array(B_xi_i))

na = robot.num_actuators
print("num_actuators:", na)
dof = sum(robot.V_dof.reshape(-1))
print("total dof:", dof)

q0 = jnp.zeros((dof,))
q0dot = jnp.zeros((dof,))
L_cum = jax.device_get(robot.V_L_cum)
total_length = float(L_cum[-1])
s_end = float(total_length)
g_end = robot.forward_kinematics(q0, s_end)

J_end = robot.jacobian_bodyframe(q0, s_end)
M = robot.inertia_matrix(q0)
K = robot.stiffness_matrix()
F = robot.gravitational_force(q0)
D = robot.damping_matrix()
B = robot.actuation_matrix(q0)
u = jnp.asarray([-1, -0.00], dtype=q0.dtype)  
C = robot.coriolis_matrix(q0, q0dot)
# #THE INITIAL CONDITION CAN BE CHANGED WITH G_INI,G_INIT e G0 IN CORE
tau = robot.actuation_force(q0, u)


# print("L_cum:", L_cum)  
# print("total length:", total_length)
print("s_end:", s_end)
print("g(s_end) SE(3):\n", g_end)
# print("Jacobian at end (6 x dof):\n", J_end)
# print("Inertia matrix:", M)
# print("Stiffness matrix:", K)
# print("Gravitational force:", F)
# print("Damping matrix:", D)
# print("Actuation matrix:", B )
# print("Coriolis matrix:", C)
# print("Actuation matrix shape:", B.shape)
# print("Input u:", u)
# print("Input u shape:", u.shape)
# print("Actuation force (tau):", tau)
# print("Actuation force shape:", tau.shape)


# y = jnp.concatenate([q0, q0dot])
tau_ext = 0* jnp.ones((dof,))
# ydot = robot.forward_dynamics(0.0, y, actuation_args=(u, tau_ext))
# print("ydot (q0dot, q0ddot):", ydot)

# res = solve_equilibrium(robot, u, q0)
# print("q* =", res.value)


# =====================================================
# Simulation upon time
# =====================================================

# Plot the initial configuration
curve = draw_robot_curve(robot, q0)
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "3d"})
ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], lw=4, color="blue")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("Initial configuration")
ax.axis("equal")
plt.show()

# Simulation time parameters
t0 = 0.0
t1 = 2.0
dt = 1e-4
ts, q_ts, qd_ts = robot.resolve_upon_time(
    q0=q0,
    qd0=q0dot,
    u=u,
    t0=t0,
    t1=t1,
    dt=dt,
    tau_ext=tau_ext,
    max_steps=None,
)
print(f"Simulation completed with {len(ts)} time steps.")
# =====================================================
# End-effector position upon time
# =====================================================
forward_kinematics_end_effector = jax.jit(
    partial(
        robot.forward_kinematics,
        s=jnp.sum(robot.V_L),  # end-effector position
    )
)
g_ee_ts = jax.vmap(forward_kinematics_end_effector)(q_ts)


### MARKERS POSITIONS UPON TIME ###
forward_kinematics_marker_1 = jax.jit(
    partial(
        robot.forward_kinematics,
        s=0.1291,  
    )
)
g_marker_1_ts = jax.vmap(forward_kinematics_marker_1)(q_ts)
p_marker_1_ts = g_marker_1_ts[:, :3, 3] + g_marker_1_ts[:, :3, :3] @ jnp.array([0.0, 0.0, 0.025]) 


forward_kinematics_marker_2 = jax.jit(
    partial(
        robot.forward_kinematics,
        s=0.2195,  
    )
)
g_marker_2_ts = jax.vmap(forward_kinematics_marker_2)(q_ts)
p_marker_2_ts = g_marker_2_ts[:, :3, 3] + g_marker_2_ts[:, :3, :3] @ jnp.array([0.0, 0.0, -0.021]) 


forward_kinematics_marker_3 = jax.jit(
    partial(
        robot.forward_kinematics,
        s=0.2800,  
    )
)
g_marker_3_ts = jax.vmap(forward_kinematics_marker_3)(q_ts)
p_marker_3_ts = g_marker_3_ts[:, :3, 3] + g_marker_3_ts[:, :3, :3] @ jnp.array([0.0, 0.0, 0.02]) 


forward_kinematics_marker_4 = jax.jit(
    partial(
        robot.forward_kinematics,
        s=jnp.sum(robot.V_L), 
    )
)
g_marker_4_ts = jax.vmap(forward_kinematics_marker_4)(q_ts)
p_marker_4_ts = g_marker_4_ts[:, :3, 3] + g_marker_4_ts[:, :3, :3] @ jnp.array([0.008, 0.0, 0.0]) 







plt.figure()
plt.plot(ts, g_ee_ts[:, 0, 3], label="End-effector x [m]")
plt.plot(ts, g_ee_ts[:, 1, 3], label="End-effector y [m]")
plt.plot(ts, g_ee_ts[:, 2, 3], label="End-effector z [m]")
plt.xlabel("Time [s]")
plt.ylabel("End-effector position [m]")
plt.legend()
plt.grid(True)
plt.box(True)
plt.tight_layout()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
p = ax.scatter(
    g_ee_ts[:, 0, 3], g_ee_ts[:, 1, 3], g_ee_ts[:, 2, 3], c=ts, cmap="viridis"
)
ax.axis("equal")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("End-effector trajectory (3D)")
fig.colorbar(p, ax=ax, label="Time [s]")
plt.show()

# =====================================================
# Plot the robot configuration upon time
# =====================================================
animate_robot_matplotlib(
    robot,
    t_list=ts,  # shape (T,)
    q_list=q_ts,  # shape (T, DOF)
    interval=100,  # ms
    slider=True,
)
