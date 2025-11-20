__all__ = ["DynamicalSystem"]
import equinox as eqx
import jax
from jax import Array, jit, vmap
from jax import numpy as jnp
from typing import Any, Callable, Optional, Tuple

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

from soromox.systems.system_state import SystemState


class DynamicalSystem(eqx.Module):
    num_actuators: int = eqx.field(static=True)  # Number of actuators

    def resolve_upon_time(
        self,
        initial_state: SystemState,
        controller: Optional[
            Callable[[SystemState], Tuple[Array, Optional[Any]]]
        ] = None,
        tau_ext: Optional[Array] = None,
        t1: Optional[float] = 10.0,
        dt: Optional[float] = 1e-4,
        save_ts: Optional[Array] = None,
        save_dt: Optional[float] = 0.01,
        solver: Optional[AbstractSolver] = Tsit5(),
        stepsize_controller: Optional[AbstractStepSizeController] = ConstantStepSize(),
        max_steps: Optional[int] = None,
    ) -> SystemState:
        """
        Resolve the system dynamics over time using Diffrax.

        Args:
            initial_state: Dataclass holding the initial time, system state vector (y),
                optional actuation (u), and optional controller state.
            controller: Callable that accepts a `SystemState` and returns a tuple
                `(u_control, control_state_dot)`. `u_control` is added to the base
                actuation from the initial state.
            tau_ext: External forces/torques applied to the system.
            t1: Final time of the simulation, included in the saved trajectory.
            dt: Time step for the solver.
            save_ts: Explicit time points to be saved in the output. Must be within
                [initial_state.t, t1]. Falls back to `save_dt` if None.
            save_dt: Time interval at which to save the solution when `save_ts` is
                not provided.
            solver: Solver to use for the ODE integration.
            stepsize_controller: Stepsize controller for the solver.
            max_steps: Maximum number of steps for the solver.

        Returns:
            SystemState PyTree containing time samples, system state trajectory,
            actuation at each saved step, and (optionally) the controller state
            trajectory. Each leaf has a leading time dimension aligned with
            `save_ts`.
        """
        base_u = initial_state.u
        if base_u is None:
            base_u = jnp.zeros((self.num_actuators,))

        y0 = initial_state.y  # initial state vector

        controller_state0 = initial_state.control_state
        track_control_state = controller_state0 is not None
        zero_control_state_dot = (
            None
            if not track_control_state
            else jnp.zeros_like(controller_state0)
            if isinstance(controller_state0, Array)
            else jax.tree_util.tree_map(jnp.zeros_like, controller_state0)
        )

        t0 = float(initial_state.t)

        if save_ts is None:
            assert save_dt is not None, "Either save_ts or save_dt must be provided."
            assert save_dt > 0.0, "save_dt must be positive."
            assert save_dt >= dt, "save_dt must be greater than or equal to the simulation dt."
            save_ts = jnp.arange(t0, t1 + save_dt, save_dt)


        if controller is not None:

            @jit
            def closed_loop_forward_dynamics(
                t: Array, ode_state, actuation_args: Tuple[Array, Array]
            ):
                """
                Evaluate the controller at (t, y[, control_state]) and apply it to the forward dynamics.
                """
                base_u_val, base_tau_ext = actuation_args
                if track_control_state:
                    y, ctrl_state = ode_state
                else:
                    y = ode_state
                    ctrl_state = None

                system_state = SystemState(t=t, y=y, u=base_u_val, control_state=ctrl_state)
                u_control, control_state_dot = controller(system_state)
                if control_state_dot is not None and not track_control_state:
                    raise ValueError(
                        "Controller returned a control_state derivative but no initial control_state was provided."
                    )
                u_total = base_u_val + u_control
                yd = self.forward_dynamics(t, y, (u_total, base_tau_ext))
                if track_control_state:
                    control_state_dot = (
                        control_state_dot if control_state_dot is not None else zero_control_state_dot
                    )
                    return yd, control_state_dot
                return yd

            forward_dynamics = closed_loop_forward_dynamics
        else:

            @jit
            def open_loop_forward_dynamics(
                t: Array, ode_state, actuation_args: Tuple[Array, Array]
            ):
                base_u_val, base_tau_ext = actuation_args
                if track_control_state:
                    y, _ = ode_state
                else:
                    y = ode_state
                yd = self.forward_dynamics(t, y, (base_u_val, base_tau_ext))
                if track_control_state:
                    return yd, zero_control_state_dot
                return yd

            forward_dynamics = open_loop_forward_dynamics

        term = ODETerm(forward_dynamics)
        saveat = SaveAt(ts=save_ts)  # Save at specified time points

        y0_for_solver = (y0, controller_state0) if track_control_state else y0

        sol = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1 + dt,
            dt0=dt,
            y0=y0_for_solver,
            args=(base_u, tau_ext),
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )

        ts = sol.ts
        if track_control_state:
            y_out, controller_state_out = sol.ys
        else:
            y_out = sol.ys
            controller_state_out = None

        # Compute controller outputs on the saved trajectory
        if controller is None:
            us = jnp.broadcast_to(base_u, (ts.shape[0], base_u.shape[0]))
        else:
            if track_control_state:
                u_controls = vmap(
                    lambda t_i, y_i, c_i: controller(
                        SystemState(t=t_i, y=y_i, u=base_u, control_state=c_i)
                    )[0]
                )(ts, y_out, controller_state_out)
            else:
                u_controls = vmap(
                    lambda t_i, y_i: controller(SystemState(t=t_i, y=y_i, u=base_u, control_state=None))[0]
                )(ts, y_out)
            us = u_controls + base_u

        return SystemState(t=ts, y=y_out, u=us, control_state=controller_state_out)
