"""Simulate a three-segment helix PCS with evenly distributed distal tendons."""

import argparse

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import Tsit5

from soromox.rendering import Open3DRenderer
from soromox.systems import HelixPCS, SystemState


def build_active_tendon_routing(
    routing_radius: float, num_tendons: int, attachment_segment_idx: int
) -> dict[str, jax.Array]:
    """Create zero-slope tendons equally distributed in the local y-z plane."""
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, num_tendons, endpoint=False)
    return {
        "ry": routing_radius * jnp.cos(angles),
        "rz": routing_radius * jnp.sin(angles),
        "my": jnp.zeros((num_tendons,), dtype=jnp.float64),
        "mz": jnp.zeros((num_tendons,), dtype=jnp.float64),
        "idx_seg_att": attachment_segment_idx
        * jnp.ones((num_tendons,), dtype=jnp.int32),
    }


def build_robot() -> HelixPCS:
    """Construct a trimmed-helicoid-inspired three-segment robot."""
    num_segments = 3
    L = jnp.array([0.16, 0.16, 0.16], dtype=jnp.float64)
    params = {
        "p0": jnp.array([jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0]),
        "L": L,
        "r": 0.03,
        "g": jnp.array([0.0, 0.0, 9.81], dtype=jnp.float64),
        "D": 1e-7
        * jnp.diag(
            (
                jnp.repeat(
                    jnp.array([[1.0, 1.0, 1.0, 1e3, 1e3, 1e3]], dtype=jnp.float64),
                    num_segments,
                    axis=0,
                )
                * L[:, None]
            ).flatten()
        ),
        "w": 0.012,
        "t": 0.002,
    }

    routing_radius = 0.75 * float(params["r"])
    active_tendon_routing_params = build_active_tendon_routing(
        routing_radius=routing_radius,
        num_tendons=4,
        attachment_segment_idx=num_segments - 1,
    )

    return HelixPCS(
        params=params,
        active_tendon_routing_params=active_tendon_routing_params,
    )


def simulate(robot: HelixPCS):
    """Run an open-loop dynamic rollout for the helix robot.

    The current helix parameterization remains numerically stiff under gravity and
    under nonzero tendon actuation, so this example uses a gravity-free equilibrium
    rollout to demonstrate the standard `rollout_to` simulation path used by the
    other PCS examples.
    """
    robot = robot.update_params({"g": jnp.zeros((3,), dtype=jnp.float64)})
    q0 = jnp.zeros((robot.num_dofs,), dtype=jnp.float64)
    q0 = q0.at[2::6].set(0.2)  # add some initial curvature to break symmetry
    print("Simulating with initial configuration q0 =\n", q0)
    qd0 = jnp.zeros_like(q0)
    u = jnp.zeros((robot.num_actuators,), dtype=jnp.float64)
    # u = 1e-6 * jnp.array(
    #     [-1.0, -1.0, -1.0, -1.0, -1.0], dtype=jnp.float64
    # )  # actuate one tendon to break symmetry

    initial_state = SystemState(t=0.0, y=jnp.concatenate([q0, qd0]))
    trajectory = robot.rollout_to(
        initial_state=initial_state,
        u=u,
        t1=1.5,
        solver_dt=5e-5,
        save_dt=0.001,
        solver=Tsit5(),
        max_steps=None,
    )
    q_ts, qd_ts = jnp.split(trajectory.y, 2, axis=1)
    return robot, trajectory.t, q_ts, qd_ts, u


def main(
    *, render_open3d: bool = False, record_path: str = "videos/helix_pcs_open3d.mp4"
):
    robot, ts, q_ts, qd_ts, u = simulate(build_robot())

    q0 = q_ts[0]
    qf = q_ts[-1]
    g_tip_0 = robot.forward_kinematics(q0, robot.length)
    g_tip_f = robot.forward_kinematics(qf, robot.length)

    print("num_segments =", robot.num_segments)
    print("num_actuators =", robot.num_actuators)
    print("segment_lengths =", robot.L)
    print("active_tendon_routing_params =")
    for key, value in robot.active_tendon_routing_params.items():
        print(f"  {key}: {value}")
    print("gravity =", robot.g[3:])
    print("u =", u)
    print("A(q0).shape =", robot.actuation_matrix(q0).shape)
    print("initial_tip_position =", g_tip_0[:3, 3])
    print("final_tip_position =", g_tip_f[:3, 3])
    print("final_active_tendon_lengths =", robot.active_tendon_length(qf))
    print("max_abs_q =", jnp.max(jnp.abs(q_ts)))
    print("max_abs_qd =", jnp.max(jnp.abs(qd_ts)))
    print("saved_time_steps =", ts.shape[0])

    strain_labels = [
        r"$\kappa_{x}$",
        r"$\kappa_{y}$",
        r"$\kappa_{z}$",
        r"$\sigma_{x}$",
        r"$\sigma_{y}$",
        r"$\sigma_{z}$",
    ]
    plt.figure()
    for segment_idx in range(robot.num_segments):
        for strain_idx, strain_label in enumerate(strain_labels):
            q_idx = 6 * segment_idx + strain_idx
            plt.plot(
                ts,
                q_ts[:, q_idx],
                label=rf"{strain_label}$^{{({segment_idx + 1})}}$",
            )
    plt.xlabel("Time [s]")
    plt.ylabel("Configuration")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if not render_open3d:
        return

    if Open3DRenderer is None:
        print("Open3DRenderer is unavailable. Install with `pip install open3d`.")
        return

    renderer = Open3DRenderer(robot)

    # render the initial configuration
    renderer.show(q0)

    renderer.render_sequence(
        ts=ts[::10],
        q_ts=q_ts[::10],
        render_tendons=True,
        record_path=record_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--open3d",
        action="store_true",
        help="Render the simulated trajectory with Open3D.",
    )
    parser.add_argument(
        "--record-path",
        default="videos/helix_pcs_open3d.mp4",
        help="Optional Open3D recording target used when --open3d is set.",
    )
    args = parser.parse_args()
    main(render_open3d=args.open3d, record_path=args.record_path)
