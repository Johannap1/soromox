"""Diffrax-compatible numerical integrators for Soromox state layouts."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from diffrax import (
    RESULTS,
    AbstractSolver,
    AbstractTerm,
    LocalLinearInterpolation,
)

from soromox.systems.dynamical_system import DynamicalSystem
from soromox.systems.system_state import SystemState


class SemiImplicitEuler(AbstractSolver):
    """Kick-drift Euler for a Soromox ``[q, qd, auxiliary]`` state.

    Diffrax's generic :class:`diffrax.SemiImplicitEuler` requires a separable
    two-term vector field. Soromox dynamics generally include
    velocity-dependent forces, so this solver evaluates the full state
    derivative at ``(q_n, qd_n)`` and applies

    ``qd_{n+1} = qd_n + dt * qdd_n`` and
    ``q_{n+1} = retract(q_n, dt * qd_{n+1})``.

    The solver is intended for :class:`soromox.systems.DynamicalSystem`
    rollouts. It uses the supplied system's state helpers rather than splitting
    the state in half, so unequal floating-base configuration and velocity
    dimensions, manifold retraction, and trailing system auxiliary states are
    supported. Diffrax-level control and environment state leaves are advanced
    with explicit Euler.

    Attributes:
        system: Soromox system that supplies ``split_state``, ``pack_state``,
            and ``retract_configuration``.
    """

    system: DynamicalSystem
    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    def order(self, terms: AbstractTerm) -> int:
        """Return the deterministic convergence order.

        Args:
            terms: Diffrax term describing the vector field. The order is one
                for every compatible term.

        Returns:
            The deterministic convergence order, which is one.
        """
        del terms
        return 1

    def init(
        self,
        terms: AbstractTerm,
        t0: Any,
        t1: Any,
        y0: Any,
        args: Any,
    ) -> None:
        """Initialize the solver state.

        Args:
            terms: Diffrax term describing the vector field.
            t0: Initial time of the integration interval.
            t1: Final time of the integration interval.
            y0: Initial Diffrax state.
            args: Additional arguments supplied to the vector field.

        Returns:
            ``None`` because the method has no persistent solver state.
        """
        del terms, t0, t1, y0, args
        return None

    def step(
        self,
        terms: AbstractTerm,
        t0: Any,
        t1: Any,
        y0: Any,
        args: Any,
        solver_state: Any,
        made_jump: Any,
    ) -> tuple[Any, None, dict[str, Any], None, RESULTS]:
        """Advance the state by one kick--drift step.

        Args:
            terms: Diffrax term describing the vector field.
            t0: Start time of the step.
            t1: End time of the step.
            y0: Diffrax state at ``t0``.
            args: Additional arguments supplied to the vector field.
            solver_state: Persistent solver state. This method does not use it.
            made_jump: Whether the solution jumped at ``t0``. This method does
                not require special jump handling.

        Returns:
            A Diffrax step tuple containing the state at ``t1``, no local error
            estimate, linear-interpolation data, no persistent solver state,
            and a successful result code.
        """
        del solver_state, made_jump
        control = terms.contr(t0, t1)
        increment = terms.vf_prod(t0, y0, args, control)
        y1 = jax.tree.map(
            lambda initial, delta: None if initial is None else initial + delta,
            y0,
            increment,
            is_leaf=lambda value: value is None,
        )

        q0, qd0, auxiliary0 = self.system.split_state(y0.y)
        _, qd_increment, auxiliary_increment = self.system.split_state(increment.y)
        qd1 = qd0 + qd_increment
        q1 = self.system.retract_configuration(q0, (t1 - t0) * qd1)
        auxiliary1 = auxiliary0 + auxiliary_increment
        primary_state1 = self.system.pack_state(q1, qd1, auxiliary1)
        y1 = dataclasses.replace(y1, y=primary_state1)

        dense_info = {"y0": y0, "y1": y1}
        return y1, None, dense_info, None, RESULTS.successful

    def func(self, terms: AbstractTerm, t0: Any, y0: Any, args: Any) -> Any:
        """Evaluate the underlying vector field.

        Args:
            terms: Diffrax term describing the vector field.
            t0: Time at which to evaluate the vector field.
            y0: Diffrax state at ``t0``.
            args: Additional arguments supplied to the vector field.

        Returns:
            Vector-field evaluation with the same PyTree structure as ``y0``.
        """
        return terms.vf(t0, y0, args)


class IMEXEuler(AbstractSolver):
    """Implicit-explicit Euler for a Soromox ``[q, qd, auxiliary]`` state.

    Treats linear stiffness and damping implicitly and everything else
    (inertia, Coriolis, gravity, actuation, nonlinear elastic residual)
    explicitly, then retracts the configuration off the new velocity:

    ``(M + h*D + h**2*K) @ qd_{n+1} = M @ qd_n + h*(rhs_explicit - K @ q_n)``
    ``q_{n+1} = retract(q_n, h*qd_{n+1})``

    Unlike ``SemiImplicitEuler``, this reads ``dynamics_terms``,
    ``stiffness_matrix``, ``damping_matrix``, ``elastic_force``, and
    ``actuation_force`` off ``system`` directly instead of going through
    ``forward_dynamics``, so it only supports open-loop rollouts and systems
    with no auxiliary state. For a floating base, the implicit stiffness term
    only acts on the internal coordinates (zero-padded to ``num_velocities``),
    which is exact as long as ``stiffness_matrix``/``damping_matrix`` have a
    zero base block -- true for every system checked so far, not guaranteed
    by the interface.

    Attributes:
        system: Soromox system supplying state and dynamics-term helpers.
    """

    system: DynamicalSystem
    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    def order(self, terms: AbstractTerm) -> int:
        """Return the deterministic convergence order, which is one."""
        del terms
        return 1

    def init(self, terms: AbstractTerm, t0: Any, t1: Any, y0: Any, args: Any) -> None:
        """No persistent solver state is needed."""
        del terms, t0, t1, y0, args
        return None

    def step(
        self,
        terms: AbstractTerm,
        t0: Any,
        t1: Any,
        y0: Any,
        args: Any,
        solver_state: Any,
        made_jump: Any,
    ) -> tuple[Any, None, dict[str, Any], None, RESULTS]:
        """Advance the state by one implicit-explicit Euler step.

        Bypasses ``terms``: ``args`` already carries the open-loop actuation
        tuple ``rollout_to`` builds, so the linear system is assembled
        directly from ``self.system``'s dynamics-term helpers.
        """
        del terms, solver_state, made_jump
        dt = t1 - t0

        y = self.system.project_state(y0.y)
        q0, qd0, auxiliary0 = self.system.split_state(y)

        u, base_tau_ext, environment_model, zero_environment_state_dot = args
        track_environment_state = zero_environment_state_dot is not None
        tau_environment = None
        environment_state_dot = zero_environment_state_dot
        if environment_model is not None:
            system_state = SystemState(
                t=t0,
                y=y,
                u=u,
                control_state=None,
                environment_state=y0.environment_state,
            )
            tau_environment, environment_state_dot = environment_model(system_state)
            if environment_state_dot is not None and not track_environment_state:
                raise ValueError(
                    "Environment model returned an environment_state derivative but no initial environment_state was provided."
                )
            environment_state_dot = (
                environment_state_dot
                if environment_state_dot is not None
                else zero_environment_state_dot
            )
        if tau_environment is None:
            tau_ext = base_tau_ext
        elif base_tau_ext is None:
            tau_ext = tau_environment
        else:
            tau_ext = base_tau_ext + tau_environment
        if tau_ext is None:
            tau_ext = jnp.zeros_like(qd0)

        _, q_internal = self.system.split_configuration(q0)
        q_embedded = jnp.concatenate(
            [
                jnp.zeros((self.system.num_base_velocities,), dtype=q_internal.dtype),
                q_internal,
            ]
        )

        M, Cqd, G = self.system.dynamics_terms(q0, qd0)
        K = self.system.stiffness_matrix()
        D = self.system.damping_matrix(q0)
        tau_u = self.system.actuation_force(q0, u, qd=qd0)
        elastic_nonlinear = self.system.elastic_force(q0) - K @ q_embedded

        rhs_explicit = tau_u + tau_ext - Cqd - G - elastic_nonlinear
        a = M + dt * D + dt**2 * K
        b = M @ qd0 + dt * (rhs_explicit - K @ q_embedded)
        qd1 = jnp.linalg.solve(a, b)

        q1 = self.system.retract_configuration(q0, dt * qd1)
        if self.system.num_auxiliary_states == 0:
            auxiliary1 = auxiliary0
        else:
            derivative = self.system.forward_dynamics(t0, y, (u, tau_ext))
            _, _, auxiliary_dot = self.system.split_state(derivative)
            auxiliary1 = auxiliary0 + dt * auxiliary_dot
        primary_state1 = self.system.pack_state(q1, qd1, auxiliary1)

        environment_state1 = (
            y0.environment_state
            if environment_state_dot is None
            else jax.tree_util.tree_map(
                lambda e, d: e + dt * d, y0.environment_state, environment_state_dot
            )
        )
        y1 = dataclasses.replace(
            y0, y=primary_state1, environment_state=environment_state1
        )

        dense_info = {"y0": y0, "y1": y1}
        return y1, None, dense_info, None, RESULTS.successful

    def func(self, terms: AbstractTerm, t0: Any, y0: Any, args: Any) -> Any:
        """Evaluate the underlying vector field."""
        return terms.vf(t0, y0, args)


__all__ = ["IMEXEuler", "SemiImplicitEuler"]
