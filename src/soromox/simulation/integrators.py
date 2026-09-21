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

from soromox.execution.mechanics import MechanicalState
from soromox.systems.soft_robot import SoftRobot


class SemiImplicitEuler(AbstractSolver):
    """Kick-drift Euler for a Soromox ``[q, qd, auxiliary]`` state.

    Diffrax's generic :class:`diffrax.SemiImplicitEuler` requires a separable
    two-term vector field. Soromox dynamics generally include
    velocity-dependent forces, so this solver evaluates the full state
    derivative at ``(q_n, qd_n)`` and applies

    ``qd_{n+1} = qd_n + dt * qdd_n`` and
    ``q_{n+1} = retract(q_n, dt * qd_{n+1})``.

    The solver is intended for :class:`soromox.systems.SoftRobot`
    rollouts. It uses the supplied system's state helpers rather than splitting
    the state in half, so unequal floating-base configuration and velocity
    dimensions, manifold retraction, and trailing system auxiliary states are
    supported. Diffrax-level control and environment state leaves are advanced
    with explicit Euler.

    Attributes:
        system: Soft robot system that supplies ``split_state``, ``pack_state``,
            and ``retract_configuration``.
    """

    system: SoftRobot
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

    Unlike ``SemiImplicitEuler``, this reads ``mechanics_terms`` (mass matrix
    and force terms, without the inertia solve) plus ``stiffness_matrix`` and
    ``damping_matrix`` off ``system`` directly instead of going through
    ``forward_dynamics``, so it can keep the linear stiffness and damping
    terms out of ``mechanics_terms``' explicit force sum. ``mechanics_terms``
    uses the same fused-or-fallback Warp dispatch as ``forward_dynamics`` on
    systems that support it (GVS, PCS, PlanarPCS), so this solver is not
    slower than ``SemiImplicitEuler`` on those systems. It resolves ``args``
    via ``DynamicalSystem._resolve_actuation``, the same helper the open- and
    closed-loop rollout wrappers use, so both rollout modes are supported.
    For a floating base, the implicit stiffness term
    only acts on the internal coordinates (zero-padded to ``num_velocities``),
    which is exact as long as ``stiffness_matrix``/``damping_matrix`` have a
    zero base block -- true for every system checked so far, not guaranteed
    by the interface.

    Attributes:
        system: Soromox system supplying state and dynamics-term helpers.
    """

    system: SoftRobot
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

        Bypasses ``terms``: ``args`` already carries the actuation tuple
        ``rollout_to``/``rollout_closed_loop_to`` build, so the linear system
        is assembled directly from ``self.system``'s dynamics-term helpers.
        """
        del terms, solver_state, made_jump
        dt = t1 - t0

        y = self.system.project_state(y0.y)
        q0, qd0, auxiliary0 = self.system.split_state(y)

        u, tau_ext, control_state_dot, environment_state_dot = (
            self.system._resolve_actuation(t0, y, y0, args)
        )
        if tau_ext is None:
            tau_ext = jnp.zeros_like(qd0)

        _, q_internal = self.system.split_configuration(q0)
        q_embedded = jnp.concatenate(
            [
                jnp.zeros((self.system.num_base_velocities,), dtype=q_internal.dtype),
                q_internal,
            ]
        )

        state = MechanicalState(configuration=q0, velocity=qd0, auxiliary=auxiliary0)
        terms = self.system.mechanics_terms(state, u, tau_ext)
        M = terms.mass_matrix
        K = self.system.stiffness_matrix()
        D = self.system.damping_matrix(q0)
        elastic_nonlinear = terms.forces.elastic - K @ q_embedded

        rhs_explicit = (
            terms.forces.actuation
            + tau_ext
            - terms.forces.convective
            - terms.forces.gravity
            - elastic_nonlinear
        )
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
        control_state1 = (
            y0.control_state
            if control_state_dot is None
            else jax.tree_util.tree_map(
                lambda c, d: c + dt * d, y0.control_state, control_state_dot
            )
        )
        y1 = dataclasses.replace(
            y0,
            y=primary_state1,
            environment_state=environment_state1,
            control_state=control_state1,
        )

        dense_info = {"y0": y0, "y1": y1}
        return y1, None, dense_info, None, RESULTS.successful

    def func(self, terms: AbstractTerm, t0: Any, y0: Any, args: Any) -> Any:
        """Evaluate the underlying vector field."""
        return terms.vf(t0, y0, args)


class IMEXEulerImplicitContact(AbstractSolver):
    """IMEX Euler with a single tip/y-direction contact force against a
    movable point-mass body, both folded implicitly into their respective
    linear systems instead of left on the explicit right-hand side / the
    generic explicit `environment_state` update used elsewhere.

    Deliberately narrow in scope: contact is evaluated only at the rod's
    very tip, only along the global y-axis, with no friction, and only a
    single contact point against a single point-mass body. The rod/body
    coupling itself is staggered (Gauss-Seidel: the rod is solved first
    using the body's start-of-step position, then the body is solved using
    the rod's just-updated tip position), not a fully joint implicit solve
    of both together.

    The body is tracked via the standard ``environment_state`` field (a
    ``(position, velocity)`` pair) so it is interoperable with
    ``rollout_to``/``rollout_closed_loop_to`` -- but note this solver
    implements its own implicit update for ``environment_state`` here,
    not the generic explicit ``e + dt * d`` rule
    ``DynamicalSystem._resolve_actuation`` callers use elsewhere. That
    generic rule was verified empirically (not just by analogy to the
    rod's own explicit-contact instability) to silently inject unphysical
    momentum into a light body at the instant contact begins, without
    ever producing a non-finite value -- a plain finiteness check would
    not catch it.

    Derivation for the rod's own contact term: linearizing the contact
    force tau_contact(q) around q0, tau_contact(q_k+1) ~= tau_contact(q0)
    + K_contact @ (q_k+1 - q0), with K_contact = d(tau_contact)/dq =
    -stiffness * outer(g, g) while in contact (g = d(tip_y)/dq, dropping
    the geometric/curvature term d(g)/dq -- the same simplification the
    rod's own K already makes for its nonlinear elastic term).
    Substituting q_k+1 = q0 + dt*qd_k+1 and moving the qd_k+1-dependent
    piece to the left-hand side adds +dt**2 * stiffness * outer(g, g) to
    the matrix being inverted. The body's own contact reaction is folded
    in the same way, specialized to its single degree of freedom.

    Fixed-base systems only, for now: K_contact = outer(g, g) is sized by
    num_coordinates (g = d(tip_y)/dq), but the matrix it is added to (K, D,
    M) is sized by num_velocities. For fixed-base systems these coincide,
    but for floating-base systems they do not (the configuration carries a
    redundant 4th quaternion component with no matching velocity
    component), which raises a shape-mismatch error rather than silently
    producing a wrong answer -- verified empirically while adding this
    class's regression tests, not merely a hypothetical gap.

    Attributes:
        system: Soromox system supplying state and dynamics-term helpers.
    """

    system: SoftRobot
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
        """Advance the rod and a movable contact body by one implicit-contact step.

        Args:
            terms: Unused; ``args`` already carries the actuation and
                contact-body parameters directly.
            t0: Start time of the step.
            t1: End time of the step.
            y0: Rollout state; ``y0.environment_state`` must be a
                ``(position, velocity)`` pair for the contact body.
            args: Tuple ``(u, body_mass, body_damping, stiffness)``. ``u``
                is the actuation vector; ``body_mass``/``body_damping``/
                ``stiffness`` describe the point-mass body tracked in
                ``y0.environment_state``.
            solver_state: Unused; this method has no persistent state.
            made_jump: Unused; no special jump handling is required.
        """
        del terms, solver_state, made_jump
        dt = t1 - t0
        u, body_mass, body_damping, stiffness = args

        y = self.system.project_state(y0.y)
        q0, qd0, auxiliary0 = self.system.split_state(y)
        body_y0, body_vy0 = y0.environment_state

        _, q_internal = self.system.split_configuration(q0)
        q_embedded = jnp.concatenate(
            [
                jnp.zeros((self.system.num_base_velocities,), dtype=q_internal.dtype),
                q_internal,
            ]
        )

        def tip_y(q_):
            return self.system.forward_kinematics_tips(q_)[-1][1, 3]

        y_tip, g = jax.value_and_grad(tip_y)(q0)
        penetration = jnp.maximum(0.0, body_y0 - y_tip)
        in_contact = penetration > 0.0
        tau_contact = g * stiffness * penetration
        K_contact = jnp.where(in_contact, stiffness, 0.0) * jnp.outer(g, g)

        tau_ext = tau_contact

        state = MechanicalState(configuration=q0, velocity=qd0, auxiliary=auxiliary0)
        mechanics = self.system.mechanics_terms(state, u, tau_ext)
        M = mechanics.mass_matrix
        K = self.system.stiffness_matrix()
        D = self.system.damping_matrix(q0)
        elastic_nonlinear = mechanics.forces.elastic - K @ q_embedded

        rhs_explicit = (
            mechanics.forces.actuation
            + tau_ext
            - mechanics.forces.convective
            - mechanics.forces.gravity
            - elastic_nonlinear
        )
        a = M + dt * D + dt**2 * (K + K_contact)
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

        # Body: implicit 1-DOF backward-Euler, staggered against the rod's
        # just-updated tip position (Gauss-Seidel, not jointly implicit).
        y_tip1 = tip_y(q1)
        body_penetration = jnp.maximum(0.0, body_y0 - y_tip1)
        body_in_contact = body_penetration > 0.0
        f_y0 = stiffness * body_penetration
        body_denom = body_mass + dt * body_damping + jnp.where(
            body_in_contact, dt**2 * stiffness, 0.0
        )
        body_vy1 = (body_mass * body_vy0 - dt * f_y0) / body_denom
        body_y1 = body_y0 + dt * body_vy1

        y1 = dataclasses.replace(
            y0,
            y=primary_state1,
            environment_state=(body_y1, body_vy1),
        )

        dense_info = {"y0": y0, "y1": y1}
        return y1, None, dense_info, None, RESULTS.successful

    def func(self, terms: AbstractTerm, t0: Any, y0: Any, args: Any) -> Any:
        """Evaluate the underlying vector field."""
        return terms.vf(t0, y0, args)


__all__ = ["IMEXEuler", "IMEXEulerImplicitContact", "SemiImplicitEuler"]
