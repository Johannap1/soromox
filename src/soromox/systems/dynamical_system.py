__all__ = ["DynamicalSystem"]
import equinox as eqx
from jax import Array, jit, lax, vmap
from jax import numpy as jnp
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Type, Union, Optional

# Diffrax (for time integration helpers)
from diffrax import (
    diffeqsolve,
    ODETerm,
    SaveAt,
    Tsit5,
    AbstractStepSizeController,
    ConstantStepSize,
    AbstractSolver,
)


class DynamicalSystem(eqx.Module):
    num_actuators: int = eqx.field(static=True)  # Number of actuators

    def resolve_upon_time(
        self,
        q0: Array,
        qd0: Array,
        u: Optional[Array] = None,
        controller: Optional[Callable[[Array, Array, Array], Array]] = None,
        tau_ext: Optional[Array] = None,
        t0: Optional[float] = 0.0,
        t1: Optional[float] = 10.0,
        dt: Optional[float] = 1e-4,
        saveat_ts: Optional[Array] = None,
        save_dt: Optional[float] = 0.01,
        solver: Optional[AbstractSolver] = Tsit5(),
        stepsize_controller: Optional[AbstractStepSizeController] = ConstantStepSize(),
        max_steps: Optional[int] = None,
    ) -> Tuple[Array, Array, Array, Array]:
        """
        Resolve the system dynamics over time using Diffrax.

        Args:
            q0 (Array): Initial configuration (strains).
            qd0 (Array): Initial velocity (strains).
            u (Array, optional): Actuation/control input.
                Default is None (no actuation).
            controller (Callable, optional): Control function/object that takes in time, configuration, and velocity,
                and returns the actuation input. If provided, it is added to the `u` argument.
                    => controller(t: Array, q: Array, qd: Array) -> Array
            tau_ext (Array, optional): External forces/torques applied to the system.
            t0 (float, optional): Initial time of the simuation, included.
                It represents the first time step resolved by the solver. Default is 0.0.
            t1 (float, optional): Final time of the simulation, included.
                It represents the last time step resolved by the solver, so that the solution of the system at
                time t1 is included in the output (we include t1 by adding dt to the interval in diffeqsolve).
                Default is 10.0.
            dt (float, optional): Time step for the solver.
                Default is 1e-4.
            saveat_ts (Array, optional): Array of time steps to be saved in the output. Must be within t0 and t1.
                If not provided, falls back to `save_dt`.
            save_dt (float, optional): Time interval at which to save the solution.
                If saveat_ts is provided, this is ignored. Default is 0.01 (i.e., save every 10 ms).
            solver (AbstractSolver, optional): Solver to use for the ODE integration.
                Default is Tsit5() (Runge-Kutta 5(4) method).
            stepsize_controller (PIDController, optional): Stepsize controller for the solver.
                Default is ConstantStepSize().
            max_steps (int, optional): Maximum number of steps for the solver.
                Default is None (no limit).

        Returns:
            ts (Array): Time points at which the solution is saved.
            qs (Array): Configuration (strains) at the saved time points.
            qds (Array): Velocity (strains) at the saved time points.
            us (Array): Actuation applied at each saved time step.
        """
        y0 = jnp.concatenate([q0, qd0])  # Initial state vector
        if u is None:
            u = jnp.zeros((self.num_actuators,))
        if tau_ext is None:
            tau_ext = jnp.zeros((q0.shape[-1],))
        if saveat_ts is None:
            assert save_dt is not None, "Either saveat_ts or save_dt must be provided."
            assert save_dt > 0.0, "save_dt must be positive."
            assert save_dt >= dt, "save_dt must be greater than or equal to the simulation dt."
            saveat_ts = jnp.arange(t0, t1 + save_dt, save_dt)

        if controller is not None:
            @jit
            def closed_loop_forward_dynamics(
                t: Array, y: Array, actuation_args: Tuple[Array, Array]
            ) -> Array:
                """
                Evaluate controller at (t, q, qd) and apply it to the forward dynamics.
                Args:
                    t: Current time
                    y: Current state vector (configuration and velocity)
                    actuation_args: Tuple containing base actuation and external torques
                Returns:
                    yd: Time derivative of the state vector
                """
                base_u, base_tau_ext = actuation_args
                q, qd = jnp.split(y, 2)
                u_control = controller(t, q, qd)
                u = base_u + u_control
                yd = self.forward_dynamics(t, y, (u, base_tau_ext))
                return yd

            forward_dynamics = closed_loop_forward_dynamics
        else:
            forward_dynamics = self.forward_dynamics
        term = ODETerm(forward_dynamics)
        saveat = SaveAt(ts=saveat_ts)  # Save at specified time points

        sol = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1 + dt,
            dt0=dt,
            y0=y0,
            args=(u, tau_ext),
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )

        ts = sol.ts
        # Extract the configuration and velocity from the solution
        y_out = sol.ys
        qs, qds = jnp.split(y_out, 2, axis=1)

        if controller is None:
            us = jnp.broadcast_to(u, (ts.shape[0], u.shape[0]))
        else:
            u_controls = vmap(lambda t_i, q_i, qd_i: controller(t_i, q_i, qd_i))(ts, qs, qds)
            us = u_controls + u

        return ts, qs, qds, us
