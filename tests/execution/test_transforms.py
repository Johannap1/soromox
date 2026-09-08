"""Tests for shared batching and differentiation transformation rules."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from numpy.testing import assert_allclose

from soromox.execution import (
    DynamicsTerms,
    ExecutionBackend,
    MechanicalState,
    evaluate_forward_dynamics,
    evaluate_mechanics_terms,
    make_dynamics_evaluator,
    make_kinematics_evaluators,
)

jax.config.update("jax_enable_x64", True)


class _TransformProbe(eqx.Module):
    """Differentiable reference model for execution transform tests."""

    scale: Array
    backend: ExecutionBackend = eqx.field(static=True, default="warp")
    num_coordinates: int = eqx.field(static=True, default=3)
    num_velocities: int = eqx.field(static=True, default=3)
    num_actuators: int = eqx.field(static=True, default=0)
    num_auxiliary_states: int = eqx.field(static=True, default=0)

    def split_state(self, y: Array) -> tuple[Array, Array, Array]:
        """Split the equal-size probe state using explicit dimensions."""

        return (
            y[: self.num_coordinates],
            y[self.num_coordinates : self.num_coordinates + self.num_velocities],
            y[self.num_coordinates + self.num_velocities :],
        )

    def configuration_derivative(self, q: Array, qd: Array) -> Array:
        """Map probe velocities directly to coordinate derivatives."""

        del q
        return qd

    def _assemble_dynamics_terms(self, q: Array, qd: Array) -> DynamicsTerms:
        """Return nonlinear terms so both tangent paths are observable."""

        inertia = self.scale * jnp.eye(self.num_velocities, dtype=q.dtype) + jnp.outer(
            q, q
        )
        coriolis_qd = self.scale * q * qd
        gravity = jnp.sin(q) + self.scale
        return inertia, coriolis_qd, gravity

    def dynamics_terms(
        self,
        q: Array,
        qd: Array,
        *,
        backend: ExecutionBackend | None = None,
    ) -> DynamicsTerms:
        """Expose which backend the outer differentiation boundary selected."""

        del qd
        factor = self.scale if backend == "jax" else 10.0 * self.scale
        return jnp.eye(self.num_velocities), factor * jnp.sin(q), jnp.zeros_like(q)

    def elastic_force(self, q: Array) -> Array:
        """Return zero elastic force for the execution probe."""

        return jnp.zeros_like(q)

    def damping_matrix(self, q: Array) -> Array:
        """Return zero damping for the execution probe."""

        return jnp.zeros((q.size, q.size), dtype=q.dtype)

    def actuation_force(
        self,
        q: Array,
        u: Array,
        *,
        qd: Array,
        backend: ExecutionBackend | None = None,
    ) -> Array:
        """Expose automatic actuation routing in the primal calculation."""

        del u, qd
        factor = 2.0 if backend == "auto" else 0.0
        return factor * jnp.ones_like(q)

    def _solve_inertia(self, inertia: Array, rhs: Array) -> Array:
        """Solve the probe's generalized inertia system."""

        return jnp.linalg.solve(inertia, rhs)


class _KinematicsTransformProbe(eqx.Module):
    """Differentiable model for kinematics custom-JVP tests."""

    offset: Array
    backend: ExecutionBackend = eqx.field(static=True, default="warp")
    num_coordinates: int = eqx.field(static=True, default=3)
    num_velocities: int = eqx.field(static=True, default=3)
    floating_base: bool = eqx.field(static=True, default=False)
    is_planar: bool = eqx.field(static=True, default=True)

    def _absolute_forward_kinematics(self, q: Array, s: Array) -> Array:
        """Return a pose that depends on state and a dynamic model leaf."""

        return self.offset + jnp.array([s + q[0], q[1], q[2]])

    def _forward_kinematics_jvp(
        self,
        q: Array,
        s: Array,
        qd: Array | None,
        sd: Array | None,
        *,
        model_tangent: Any = None,
    ) -> tuple[Array, Array]:
        """Differentiate the probe's model and state inputs."""

        qd = jnp.zeros_like(q) if qd is None else qd
        sd = jnp.zeros_like(s) if sd is None else sd
        pose, pose_tangent = jax.jvp(
            lambda q_, s_: self._absolute_forward_kinematics(q_, s_),
            (q, s),
            (qd, sd),
        )
        if model_tangent is not None and any(
            eqx.is_inexact_array(leaf) for leaf in jax.tree.leaves(model_tangent)
        ):
            _, model_pose_tangent = eqx.filter_jvp(
                lambda candidate: candidate._absolute_forward_kinematics(q, s),
                (self,),
                (model_tangent,),
            )
            if model_pose_tangent is not None:
                pose_tangent = pose_tangent + model_pose_tangent
        return pose, pose_tangent


def _assert_terms_close(actual: DynamicsTerms, expected: DynamicsTerms) -> None:
    """Compare the three dynamics outputs with strict FP64 tolerances."""

    for actual_term, expected_term in zip(actual, expected, strict=True):
        assert_allclose(actual_term, expected_term, rtol=1e-12, atol=1e-12)


def _assert_mechanics_terms_close(actual: Any, expected: Any) -> None:
    """Compare two MechanicalDynamicsTerms pytrees with strict FP64 tolerances."""

    actual_leaves = jax.tree_util.tree_leaves(actual)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        assert_allclose(actual_leaf, expected_leaf, rtol=1e-12, atol=1e-12)


def test_dynamics_evaluator_uses_jax_for_forward_and_reverse_derivatives() -> None:
    """Treat custom JVP as routing, not as a new analytical derivative."""

    model = _TransformProbe(scale=jnp.asarray(2.0, dtype=jnp.float64))
    q = jnp.asarray([0.1, -0.2, 0.3], dtype=jnp.float64)
    qd = jnp.asarray([-0.4, 0.5, 0.6], dtype=jnp.float64)
    tangent = jnp.asarray([0.7, -0.8, 0.9], dtype=jnp.float64)

    def execute_batch(model: _TransformProbe, q: Array, qd: Array) -> DynamicsTerms:
        del qd
        batch_size = q.shape[0]
        return (
            jnp.ones((batch_size, model.num_velocities, model.num_velocities)),
            2.0 * jnp.ones((batch_size, model.num_velocities)),
            3.0 * jnp.ones((batch_size, model.num_velocities)),
        )

    evaluate = make_dynamics_evaluator(execute_batch, family_name="test")
    expected_jvp = jax.jvp(
        lambda q_: model._assemble_dynamics_terms(q_, qd),
        (q,),
        (tangent,),
    )
    actual_jvp = jax.jvp(lambda q_: evaluate(model, q_, qd), (q,), (tangent,))
    _assert_terms_close(actual_jvp[0], expected_jvp[0])
    _assert_terms_close(actual_jvp[1], expected_jvp[1])

    expected_gradient = jax.grad(
        lambda q_: sum(
            jnp.sum(value) for value in model._assemble_dynamics_terms(q_, qd)
        )
    )(q)
    actual_gradient = jax.grad(
        lambda q_: sum(jnp.sum(value) for value in evaluate(model, q_, qd))
    )(q)
    assert_allclose(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def test_kinematics_evaluator_preserves_model_tangent() -> None:
    """Differentiate model leaves through the accelerated pose adapter."""

    model = _KinematicsTransformProbe(offset=jnp.zeros(3, dtype=jnp.float64))
    q = jnp.asarray([0.1, -0.2, 0.3], dtype=jnp.float64)
    s = jnp.asarray(0.4, dtype=jnp.float64)

    def execute_batch(
        candidate: _KinematicsTransformProbe,
        q_batch: Array,
        s_batch: Array,
        operation: str,
    ) -> Array:
        assert operation == "pose"
        return jax.vmap(
            lambda q_, s_: jax.vmap(
                lambda s__: candidate._absolute_forward_kinematics(q_, s__)
            )(s_)
        )(q_batch, s_batch)

    evaluate, _ = make_kinematics_evaluators(
        execute_batch,
        family_name="test",
        operation="pose",
    )

    def pose_sum(offset: Array, *, accelerated: bool) -> Array:
        candidate = eqx.tree_at(lambda current: current.offset, model, offset)
        pose = (
            evaluate(candidate, q, s)
            if accelerated
            else candidate._absolute_forward_kinematics(q, s)
        )
        return jnp.sum(pose)

    actual = jax.grad(lambda offset: pose_sum(offset, accelerated=True))(model.offset)
    expected = jax.grad(lambda offset: pose_sum(offset, accelerated=False))(
        model.offset
    )

    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert jnp.any(actual != 0.0)


def test_forward_dynamics_boundary_routes_derivatives_to_jax() -> None:
    """Differentiate the complete calculation without staging its Warp primal."""

    model = _TransformProbe(scale=jnp.asarray(2.0, dtype=jnp.float64))
    time = jnp.asarray(0.0, dtype=jnp.float64)
    y = jnp.linspace(-0.3, 0.4, 2 * model.num_velocities, dtype=jnp.float64)
    tangent = jnp.linspace(0.5, -0.2, y.size, dtype=jnp.float64)

    primal = evaluate_forward_dynamics(model, time, y, None, None)
    expected_primal = jnp.concatenate(
        (
            y[model.num_velocities :],
            2.0 - 10.0 * model.scale * jnp.sin(y[: model.num_velocities]),
        )
    )
    assert_allclose(primal, expected_primal, rtol=0.0, atol=0.0)

    expected_jvp = jax.jvp(
        lambda y_: evaluate_forward_dynamics(model, time, y_, None, "jax"),
        (y,),
        (tangent,),
    )
    actual_jvp = jax.jvp(
        lambda y_: evaluate_forward_dynamics(model, time, y_, None, None),
        (y,),
        (tangent,),
    )
    for actual, expected in zip(actual_jvp, expected_jvp, strict=True):
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    expected_gradient = jax.grad(
        lambda y_: jnp.sum(evaluate_forward_dynamics(model, time, y_, None, "jax"))
    )(y)
    actual_gradient = jax.grad(
        lambda y_: jnp.sum(evaluate_forward_dynamics(model, time, y_, None, None))
    )(y)
    assert_allclose(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def test_mechanics_terms_boundary_routes_derivatives_to_jax() -> None:
    """Differentiate the mechanics terms without staging their Warp primal."""

    model = _TransformProbe(scale=jnp.asarray(2.0, dtype=jnp.float64))
    q = jnp.linspace(-0.3, 0.4, model.num_velocities, dtype=jnp.float64)
    qd = jnp.linspace(0.2, -0.1, model.num_velocities, dtype=jnp.float64)
    u = jnp.zeros((model.num_actuators,), dtype=jnp.float64)
    tau_ext = jnp.linspace(-0.1, 0.1, model.num_velocities, dtype=jnp.float64)
    auxiliary = jnp.zeros((model.num_auxiliary_states,), dtype=jnp.float64)
    tangent = jnp.linspace(0.5, -0.2, q.size, dtype=jnp.float64)

    def _state(q_: Array) -> MechanicalState:
        return MechanicalState(configuration=q_, velocity=qd, auxiliary=auxiliary)

    primal = evaluate_mechanics_terms(model, _state(q), u, tau_ext, None)
    assert_allclose(
        primal.mass_matrix, jnp.eye(model.num_velocities), rtol=0.0, atol=0.0
    )
    assert_allclose(
        primal.forces.convective, 10.0 * model.scale * jnp.sin(q), rtol=0.0, atol=0.0
    )
    assert_allclose(primal.forces.actuation, 2.0 * jnp.ones_like(q), rtol=0.0, atol=0.0)

    expected_jvp = jax.jvp(
        lambda q_: evaluate_mechanics_terms(model, _state(q_), u, tau_ext, "jax"),
        (q,),
        (tangent,),
    )
    actual_jvp = jax.jvp(
        lambda q_: evaluate_mechanics_terms(model, _state(q_), u, tau_ext, None),
        (q,),
        (tangent,),
    )
    _assert_mechanics_terms_close(actual_jvp[0], expected_jvp[0])
    _assert_mechanics_terms_close(actual_jvp[1], expected_jvp[1])

    def _reduce(q_: Array, backend: ExecutionBackend | None) -> Array:
        terms = evaluate_mechanics_terms(model, _state(q_), u, tau_ext, backend)
        return jnp.sum(terms.mass_matrix) + jnp.sum(terms.forces.net_force)

    expected_gradient = jax.grad(lambda q_: _reduce(q_, "jax"))(q)
    actual_gradient = jax.grad(lambda q_: _reduce(q_, None))(q)
    assert_allclose(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
