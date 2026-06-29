"""
Simulates an Additive Manufacturing (AM) I-SUPPORT manipulator driven by
RECORDED chamber pressures exported from MATLAB (6 x M matrix sampled on
merged_time), fed as a time-varying open-loop input through the closed-loop
rollout hook.

System parameters adapted from:
Alessi, Carlo, Egidio Falotico, and Alessandro Lucantonio.
"Ablation study of a dynamic model for a 3d-printed pneumatic soft robotic arm."
IEEE Access 11 (2023): 37840-37853.
https://ieeexplore.ieee.org/abstract/document/10098800
"""

from functools import partial

## Disable warning for memory preallocation
import os
# execute this command here: export XLA_PYTHON_CLIENT_PREALLOCATE=false
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import scipy.io as sio
import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

# # Test adaptive/implicit integration
# from diffrax import Tsit5, PIDController
# from diffrax import ImplicitEuler, PIDController
# from diffrax import Kvaerno5, PIDController

jax.config.update("jax_enable_x64", True)  # double precision
from soromox.rendering import MatplotlibRenderer
from soromox.systems import ISupport, ISupportParams, SystemState

# ---- Load real actuation ----
data = sio.loadmat("amisupport_dataset/2026-06-08-11-59-18_py.mat", squeeze_me=True)

## Extract Data
poses_data  = data["poses_data"]      # (7, N, M)
markers     = data["markers"]         # (3*num_markers, M)
merged_time = np.asarray(data["merged_time"]).squeeze()
N           = int(data["N"])
M           = int(data["M"])

# Input samples
act_values = np.asarray(data["actuation_input"], dtype=np.float64)  # (6, M)
assert act_values.shape[0] == 6, act_values.shape
assert act_values.shape[1] == merged_time.shape[0], (act_values.shape, merged_time.shape)

# bar2Pa
PRESSURE_SCALE = 1.0e5
act_values = act_values * PRESSURE_SCALE

act_time     = merged_time - merged_time[0]   # zero the clock
act_time_j   = jnp.asarray(act_time)
act_values_j = jnp.asarray(act_values)        # (6, M)

# =====================================================
# Recorded-input "controller": ignores state, returns interp(t)
# =====================================================
class RecordedInput(eqx.Module):
    act_time: jax.Array        # (M,)
    act_values: jax.Array      # (6, M)

    def __call__(self, state: SystemState):
        """Interpolate the 6 recorded chamber pressures at the live solver time.
        Returns flat (6,) ordered [seg0_ch0, seg0_ch1, seg0_ch2, seg1_ch0, seg1_ch1, seg1_ch2].
        Permute the rows here if your ROS node uses a different chamber order.
        """
        t = state.t
        
        # Debug for track simulation status
        jax.debug.print("t = {t}", t=t)

        u_t = jnp.stack([
            jnp.interp(t, self.act_time, self.act_values[k, :])
            for k in range(self.act_values.shape[0])
        ])
        return u_t, None       # no control_state derivative

recorded_input = RecordedInput(act_time=act_time_j, act_values=act_values_j)

if __name__ == "__main__":
    num_segments = 2

    # Elastic modulus and poisson ratio
    E = 1.6464 * 1e6  # Elastic modulus [Pa]
    poisson_ratio = 0.5
    G = E / (
        2 * (1 + poisson_ratio)
    )  # Shear modulus from elastic modulus and poisson ratio

    segment_lengths = 190 * 1e-3 * jnp.ones((num_segments,))

    # damping coefficient
    # these values are from the paper but they seem way too large
    # gamma_t = 806  # translational damping constant [1/s]
    # gamma_r = 1.9416 * 10**(-4)  # rotational damping constant [m^2/s]
    gamma_t = 806 * 1e-3  # translational damping constant [1/s]
    gamma_r = 1.0 * 1e-3  # rotational damping constant [m^2/s]
    # Damping is specified per unit backbone length and must be integrated over
    # each segment, matching the strain-space stiffness assembly. Without this
    # length scaling the velocity term makes the two-segment fixed-step rollout
    # unnecessarily stiff and can drive the explicit solver to NaNs.
    damping_matrix = jnp.diag(
        (
            jnp.repeat(
                jnp.array([[gamma_r, gamma_r, gamma_r, gamma_t, gamma_t, gamma_t]]),
                num_segments,
                axis=0,
            )
            * segment_lengths[:, None]
        ).flatten()
    )
    params = ISupportParams(
        base_pose=jnp.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0]),
        length=segment_lengths,
        radius=35.6 * 1e-3 * jnp.ones((num_segments,)),
        density=1104 * jnp.ones((num_segments,)),
        gravity=jnp.array([0.0, 0.0, 9.81]),
        young_modulus=E * jnp.ones((num_segments,)),
        shear_modulus=G * jnp.ones((num_segments,)),
        damping_matrix=damping_matrix,
        reference_strain=jnp.tile(
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), num_segments
        ),
        chamber_inner_radius=6.39 * 1e-3 * jnp.ones((num_segments,)),
        chamber_outer_radius=7.79 * 1e-3 * jnp.ones((num_segments,)),
        chamber_distance=20 * 1e-3 * jnp.ones((num_segments,)),
        chamber_angle_offset=jnp.zeros((num_segments,)),
    )

    # ======================================================
    # Robot initialization
    # ======================================================
    robot = ISupport(params=params)

    # =====================================================
    # Initial state: undeformed reference strain, base_u = zeros
    # =====================================================
    q0 = jnp.tile(jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), num_segments)
    qd0 = jnp.zeros_like(q0)

    # Start and End time of the simulation
    t0 = 15.0
    t1 = 16.0

    initial_state = SystemState(
        t=t0,                                 # plain float (NOT jnp.array): keeps the save grid static
        y=jnp.concatenate([q0, qd0]),
        u=jnp.zeros((robot.num_actuators,)),   # base_u = 0; all input via controller
        control_state=None,                    # no control state to track
    )

    # =====================================================
    # Single closed-loop call = time-varying open-loop input
    # =====================================================
    solver_dt = 1e-4
    save_dt = 1e-4

    # fixing bug for float accumulation: [0, t1] issue
    save_ts = jnp.arange(t0, t1 + save_dt, save_dt)
    save_ts = save_ts[save_ts <= t1]

    #################################################################
    # Open-loop control law: implemented as closed_loop_to          #
    # with base_u = 0 and the function that ignores q, qdot, ...    #
    #################################################################

    trajectory = robot.rollout_closed_loop_to(
        initial_state=initial_state,
        controller=recorded_input,
        t1=t1,
        solver_dt=solver_dt,
        save_dt=save_dt,
        save_ts=save_ts,
        # solver=Tsit5(),
        # solver=ImplicitEuler(),
        # stepsize_controller=PIDController(rtol=1e-5, atol=1e-7),
        max_steps=None,
    )

    ts = trajectory.t
    q_ts, qd_ts = jnp.split(trajectory.y, 2, axis=1)
    u_ts = trajectory.u       # actuation actually applied at each save step

    # =====================================================
    # Configuration q and velocity qd upon time
    # =====================================================
    # DOF labels per segment: [kappa_x, kappa_y, kappa_z, sigma_x, sigma_y, sigma_z]
    dof_names = ["kx", "ky", "kz", "sx", "sy", "sz"]
    n_dof = q_ts.shape[1]
    labels = [f"seg{j}_{dof_names[i]}" for j in range(num_segments) for i in range(6)][:n_dof]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i in range(n_dof):
        axes[0].plot(ts, q_ts[:, i], label=labels[i])
    axes[0].set_ylabel("q (strain)")
    axes[0].set_title("Configuration q over time")
    axes[0].grid(True)
    axes[0].legend(ncol=4, fontsize=8)

    for i in range(n_dof):
        axes[1].plot(ts, qd_ts[:, i], label=labels[i])
    axes[1].set_ylabel("qd (strain rate)")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_title("Velocity qd over time")
    axes[1].grid(True)
    axes[1].legend(ncol=4, fontsize=8)

    plt.tight_layout()
    plt.show()

    # =====================================================
    # End-effector position upon time
    # =====================================================
    forward_kinematics_end_effector = jax.jit(
        partial(
            robot.forward_kinematics,
            s=jnp.sum(robot.L),  # end-effector position
        )
    )
    g_ee_ts = jax.vmap(forward_kinematics_end_effector)(q_ts)

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
    # Applied actuation (sanity check vs recorded data)
    # =====================================================
    plt.figure()
    for k in range(u_ts.shape[1]):
        plt.plot(ts, u_ts[:, k], label=f"u{k}")
    plt.xlabel("Time [s]")
    plt.ylabel("Applied pressure [Pa]")
    plt.legend(ncol=3)
    plt.grid(True)
    plt.title("Applied actuation")
    plt.tight_layout()
    plt.show()

    # # =====================================================
    # # Energy computation upon time
    # # =====================================================
    # U_ts = jax.vmap(jax.jit(partial(robot.potential_energy)))(q_ts)
    # T_ts = jax.vmap(jax.jit(partial(robot.kinetic_energy)))(q_ts, qd_ts)

    # plt.figure()
    # plt.plot(ts, U_ts, label="Potential Energy")
    # plt.plot(ts, T_ts, label="Kinetic Energy")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Energy (J)")
    # plt.legend()
    # plt.title("Energy over Time")
    # plt.grid(True)
    # plt.box(True)
    # plt.tight_layout()
    # plt.show()

    # # =====================================================
    # # Plot the robot configuration upon time
    # # =====================================================
    # renderer = MatplotlibRenderer(robot, num_points=50)
    # renderer.animate(ts=ts, q_ts=q_ts, interval=100, mode="slider")