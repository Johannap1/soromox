from collections.abc import Callable
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # double precision
import scipy as sp
from jax import Array, jacrev, jit, random
from jax import numpy as jnp

from soromox.systems import PlanarHSA, PlanarHSAParams, PlanarHSAStructure


def _repo_hsa_params_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "assets"
        / "robot_parameters"
        / "planar_hsa"
        / "fpu_control.npz"
    )


def factory_fn(
    params: PlanarHSAParams,
    strain_selector: Array,
    verbose: bool = False,
) -> tuple[Callable, Callable]:
    """
    Factory function for the planar HSA.
    Args:
        params: typed planar HSA parameters
        strain_selector: boolean array to select the strains to be activated
        verbose: flag to print additional information

    Returns:
        phi2chi_static_model_fn: function that maps motor angles to the end-effector pose
        jac_phi2chi_static_model_fn: function that computes the Jacobian between the actuation space and the task-space
    """
    robot = PlanarHSA(
        params=params,
        structure=PlanarHSAStructure(
            strain_selector=strain_selector,
        ),
    )

    def residual_fn(q: Array, phi: Array) -> Array:
        G = robot.gravitational_force(q)
        K = robot.elastic_force(q)
        alpha = robot.actuation_force(q, phi)
        res = alpha - G - K
        return jnp.square(res).mean()

    # jit the residual function
    residual_fn = jit(residual_fn)
    print("Compiling residual_fn...")
    print(residual_fn(jnp.zeros((3,)), jnp.zeros((2,))))

    # The optimizer solves grad_q residual_fn(q, phi) = 0. Differentiate this
    # stationarity condition with JAX to obtain dq/dphi through the implicit
    # function theorem, rather than finite-differencing the optimizer output.
    grad_residual_fn = jacrev(residual_fn, argnums=0)
    jac_residual_fn = jit(grad_residual_fn)
    hess_residual_fn = jit(jacrev(grad_residual_fn, argnums=0))
    mixed_residual_fn = jit(jacrev(grad_residual_fn, argnums=1))
    print("Compiling jac_residual_fn...")
    print(jac_residual_fn(jnp.zeros((3,)), jnp.zeros((2,))))

    def phi2q_static_model_fn(
        phi: Array, q0: Array = jnp.zeros((3,))
    ) -> tuple[Array, dict[str, Array]]:
        """
        A static model mapping the motor angles to the planar HSA configuration using scipy.optimize.minimize.
        Arguments:
            phi: motor angles
            q0: initial guess for the configuration

        Returns:
            q: planar HSA configuration consisting of (k_be, sigma_sh, sigma_ax)
            aux: dictionary with auxiliary data
        """
        # solve the nonlinear least squares problem
        sol = sp.optimize.minimize(
            fun=lambda q: residual_fn(q, phi).item(),
            x0=q0,
            jac=lambda q: jac_residual_fn(q, phi),
            options={"disp": True} if verbose else None,
        )
        if verbose:
            print(
                "Optimization converged after",
                sol.nit,
                "iterations with residual",
                sol.fun,
            )

        # configuration that minimizes the residual
        q = jnp.array(sol.x)

        aux = {
            "phi": phi,
            "q": q,
            "residual": sol.fun,
        }

        return q, aux

    def phi2chi_static_model_fn(
        phi: Array, q0: Array = jnp.zeros((3,))
    ) -> tuple[Array, dict[str, Array]]:
        """
        A static model mapping the motor angles to the planar end-effector pose.
        Arguments:
            phi: motor angles
            q0: initial guess for the configuration

        Returns:
            chi: end-effector pose
            aux: dictionary with auxiliary data
        """
        q, aux = phi2q_static_model_fn(phi, q0=q0)
        chi = robot.forward_kinematics_end_effector(q)
        aux["chi"] = chi
        return chi, aux

    def jac_phi2chi_static_model_fn(
        phi: Array, q0: Array = jnp.zeros((3,))
    ) -> tuple[Array, dict[str, Array]]:
        """
        Compute the Jacobian between the actuation space and the task-space.
        Arguments:
            phi: motor angles
        """
        # evaluate the static model to convert motor angles into a configuration
        q, aux = phi2q_static_model_fn(phi, q0=q0)
        # Differentiate the optimizer's first-order stationarity condition:
        # H_qq dq/dphi + H_qphi = 0.
        H_qq = hess_residual_fn(q, phi)
        H_qphi = mixed_residual_fn(q, phi)
        J_phi2q = -jnp.linalg.solve(H_qq, H_qphi)

        # Use the model's analytic end-effector Jacobian in the same pose
        # ordering returned by ``phi2chi_static_model_fn``.
        J_q2chi = robot.jacobian_end_effector(q)

        # evaluate the Jacobian between the actuation and the task-space
        J_phi2chi = J_q2chi @ J_phi2q

        return J_phi2chi, aux

    return phi2chi_static_model_fn, jac_phi2chi_static_model_fn


if __name__ == "__main__":
    num_segments = 1
    num_rods_per_segment = 2

    # activate all strains (i.e. bending, shear, and axial)
    strain_selector = jnp.ones((3 * num_segments,), dtype=bool)
    params = PlanarHSAParams.from_npz(_repo_hsa_params_path())

    # call the factory function
    phi2chi_static_model_fn, jac_phi2chi_static_model_fn = factory_fn(
        params=params,
        strain_selector=strain_selector,
    )

    phi_max = params.phi_max.flatten()

    # define initial configuration
    q0 = jnp.array([0.0, 0.0, 0.0])

    rng = random.key(seed=0)
    for i in range(10):
        match i:
            case 0:
                phi = jnp.array([0.0, 0.0])
            case 1:
                phi = jnp.array([1.0, 1.0])
            case _:
                rng, subkey = random.split(rng)
                phi = random.uniform(subkey, phi_max.shape, minval=0.0, maxval=phi_max)

        print("i", i)

        chi, aux = phi2chi_static_model_fn(phi, q0=q0)
        print("phi", phi, "q", aux["q"], "chi", chi)

        J_phi2chi, aux = jac_phi2chi_static_model_fn(phi, q0=q0)
        print("J_phi2chi:\n", J_phi2chi)
