"""JAX transformation rules shared by optional dynamics executors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from soromox.systems.execution.types import (
    AbscissaBatchedKinematicsEvaluator,
    DynamicsEvaluator,
    DynamicsModel,
    DynamicsTerms,
    ExecutionBackend,
    ForwardDynamicsModel,
    KinematicsEvaluator,
    KinematicsModel,
    KinematicsOperation,
    KinematicsResult,
)

BatchExecutor = Callable[[DynamicsModel, Array, Array], DynamicsTerms]
KinematicsBatchExecutor = Callable[
    [KinematicsModel, Array, Array, KinematicsOperation], KinematicsResult
]


@jax.custom_batching.custom_vmap
def _evaluate_forward_kinematics_primal(
    model: KinematicsModel, q: Array, s: Array
) -> Array:
    """Evaluate the scalar protected pose implementation."""

    return model._forward_kinematics(q, s)


@_evaluate_forward_kinematics_primal.def_vmap
def _evaluate_forward_kinematics_vmap(
    axis_size: int,
    in_batched: tuple[Any, bool, bool],
    model: KinematicsModel,
    q: Array,
    s: Array,
) -> tuple[Array, bool]:
    """Use the model's specialized traversal for spatial vectorization."""

    model_batched, q_batched, s_batched = in_batched
    if any(jax.tree.leaves(model_batched)):
        raise ValueError("Batching over kinematics model parameters is unsupported.")
    if not q_batched and s_batched:
        result = model._forward_kinematics_abscissa_batched(q, s)
    elif q_batched and not s_batched:
        result = jax.vmap(model._forward_kinematics, in_axes=(0, None))(q, s)
    else:
        result = jax.vmap(model._forward_kinematics)(q, s)
    return result, True


@eqx.filter_custom_jvp
def evaluate_forward_kinematics(model: KinematicsModel, q: Array, s: Array) -> Array:
    """Evaluate one forward-kinematics request with its established JVP.

    Args:
        model: System implementing the neutral kinematics contract.
        q: Generalized coordinates for one environment.
        s: One backbone coordinate.

    Returns:
        The model-specific pose at ``s``.
    """

    return _evaluate_forward_kinematics_primal(model, q, s)


@evaluate_forward_kinematics.def_jvp
def _evaluate_forward_kinematics_jvp(
    primals: tuple[KinematicsModel, Array, Array],
    tangents: tuple[Any, Array | None, Array | None],
) -> tuple[Array, Array]:
    """Route pose derivatives through the model's established JAX rule."""

    model, q, s = primals
    _model_tangent, qd, sd = tangents
    return model._forward_kinematics_jvp(q, s, qd, sd)


@jax.custom_batching.custom_vmap
def evaluate_inertial_jacobian(model: KinematicsModel, q: Array, s: Array) -> Array:
    """Evaluate one inertial Jacobian with specialized spatial batching.

    Args:
        model: System implementing the neutral kinematics contract.
        q: Generalized coordinates for one environment.
        s: One backbone coordinate.

    Returns:
        The inertial-frame Jacobian at ``s``.
    """

    return model._jacobian_inertialframe(q, s)


@evaluate_inertial_jacobian.def_vmap
def _evaluate_inertial_jacobian_vmap(
    axis_size: int,
    in_batched: tuple[Any, bool, bool],
    model: KinematicsModel,
    q: Array,
    s: Array,
) -> tuple[Array, bool]:
    """Use the model's specialized traversal for spatial vectorization."""

    model_batched, q_batched, s_batched = in_batched
    if any(jax.tree.leaves(model_batched)):
        raise ValueError("Batching over kinematics model parameters is unsupported.")
    if not q_batched and s_batched:
        result = model._jacobian_inertialframe_abscissa_batched(q, s)
    elif q_batched and not s_batched:
        result = jax.vmap(model._jacobian_inertialframe, in_axes=(0, None))(q, s)
    else:
        result = jax.vmap(model._jacobian_inertialframe)(q, s)
    return result, True


def _reference_kinematics(
    model: KinematicsModel,
    q: Array,
    s: Array,
    operation: KinematicsOperation,
) -> KinematicsResult:
    """Evaluate the requested differentiable scalar JAX kinematics result."""

    if operation == "pose":
        return evaluate_forward_kinematics(model, q, s)
    if operation == "jacobian":
        return evaluate_inertial_jacobian(model, q, s)
    return (
        evaluate_forward_kinematics(model, q, s),
        evaluate_inertial_jacobian(model, q, s),
    )


def make_kinematics_evaluators(
    execute_batch: KinematicsBatchExecutor,
    *,
    family_name: str,
    operation: KinematicsOperation,
) -> tuple[KinematicsEvaluator, AbscissaBatchedKinematicsEvaluator]:
    """Adapt a canonical ``(E,D)``/``(E,N)`` executor to public JAX semantics."""

    def slice_environment_sample(result: KinematicsResult) -> KinematicsResult:
        return jax.tree.map(lambda value: value[0, 0], result)

    def slice_environment(result: KinematicsResult) -> KinematicsResult:
        return jax.tree.map(lambda value: value[0], result)

    @jax.custom_batching.custom_vmap
    def scalar_primal(model: KinematicsModel, q: Array, s: Array) -> KinematicsResult:
        result = execute_batch(model, q[None, :], s.reshape(1, 1), operation)
        return slice_environment_sample(result)

    @scalar_primal.def_vmap
    def scalar_vmap_rule(
        axis_size: int,
        in_batched: tuple[Any, bool, bool],
        model: KinematicsModel,
        q: Array,
        s: Array,
    ) -> tuple[KinematicsResult, Any]:
        model_batched, q_batched, s_batched = in_batched
        if any(jax.tree.leaves(model_batched)):
            raise ValueError(
                f"Batching over {family_name} model parameters is not supported."
            )
        if q_batched:
            if not s_batched:
                s = jnp.broadcast_to(s, (axis_size,))
            result = execute_batch(model, q, s[:, None], operation)
            result = jax.tree.map(lambda value: value[:, 0], result)
        else:
            # Keep the spatial batch behind its own custom batching boundary.
            # An enclosing environment vmap can then merge q's mapped axis
            # into the same canonical ``(E,D)``/``(E,N)`` executor call.
            result = abscissa_batched_primal(model, q, s)
        return result, jax.tree.map(lambda _: True, result)

    @eqx.filter_custom_jvp
    def scalar_evaluator(
        model: KinematicsModel, q: Array, s: Array
    ) -> KinematicsResult:
        return scalar_primal(model, q, s)

    @scalar_evaluator.def_jvp
    def scalar_jvp(
        primals: tuple[KinematicsModel, Array, Array],
        tangents: tuple[Any, Array | None, Array | None],
    ) -> tuple[KinematicsResult, KinematicsResult]:
        model, q, s = primals
        _model_tangent, qd, sd = tangents
        if operation == "pose":
            return model._forward_kinematics_jvp(q, s, qd, sd)
        if operation == "jacobian":
            return eqx.filter_jvp(
                lambda model_, q_, s_: model_._jacobian_inertialframe(q_, s_),
                primals,
                tangents,
            )
        pose, pose_tangent = model._forward_kinematics_jvp(q, s, qd, sd)
        jacobian, jacobian_tangent = eqx.filter_jvp(
            lambda model_, q_, s_: model_._jacobian_inertialframe(q_, s_),
            primals,
            tangents,
        )
        return (pose, jacobian), (pose_tangent, jacobian_tangent)

    @jax.custom_batching.custom_vmap
    def abscissa_batched_primal(
        model: KinematicsModel, q: Array, s: Array
    ) -> KinematicsResult:
        return slice_environment(
            execute_batch(model, q[None, :], s[None, :], operation)
        )

    @abscissa_batched_primal.def_vmap
    def abscissa_batched_vmap_rule(
        axis_size: int,
        in_batched: tuple[Any, bool, bool],
        model: KinematicsModel,
        q: Array,
        s: Array,
    ) -> tuple[KinematicsResult, Any]:
        model_batched, q_batched, s_batched = in_batched
        if any(jax.tree.leaves(model_batched)):
            raise ValueError(
                f"Batching over {family_name} model parameters is not supported."
            )
        if not q_batched:
            q = jnp.broadcast_to(q, (axis_size, *q.shape))
        if not s_batched:
            s = jnp.broadcast_to(s, (axis_size, *s.shape))
        result = execute_batch(model, q, s, operation)
        return result, jax.tree.map(lambda _: True, result)

    @eqx.filter_custom_jvp
    def abscissa_batched_evaluator(
        model: KinematicsModel, q: Array, s: Array
    ) -> KinematicsResult:
        return abscissa_batched_primal(model, q, s)

    @abscissa_batched_evaluator.def_jvp
    def abscissa_batched_jvp(
        primals: tuple[KinematicsModel, Array, Array],
        tangents: tuple[Any, Array | None, Array | None],
    ) -> tuple[KinematicsResult, KinematicsResult]:
        def jax_abscissa_batched(
            model_: KinematicsModel, q_: Array, s_: Array
        ) -> KinematicsResult:
            return jax.vmap(
                lambda s_value: _reference_kinematics(model_, q_, s_value, operation)
            )(s_)

        return eqx.filter_jvp(jax_abscissa_batched, primals, tangents)

    return scalar_evaluator, abscissa_batched_evaluator


def make_dynamics_evaluator(
    execute_batch: BatchExecutor, *, family_name: str
) -> DynamicsEvaluator:
    """Adapt a batch-shaped executor to JAX scalar transformation semantics.

    The returned callable accepts one environment. Applying :func:`jax.vmap`
    does not replicate batch-one executor launches: the custom batching rule
    moves the mapped axis into a single call to ``execute_batch``. Applying
    forward- or reverse-mode differentiation replaces the forward-only executor
    with ``model._assemble_dynamics_terms`` and differentiates that JAX
    implementation. This is an execution-routing rule, not an alternative
    analytical dynamics derivative.

    Args:
        execute_batch: Family implementation accepting ``q`` and ``qd`` with
            shape ``(batch_size, num_dofs)`` and returning batched terms.
        family_name: Human-readable family name used in batching errors.

    Returns:
        A scalar-semantics dynamics evaluator with custom ``vmap`` and JVP
        behavior.
    """

    @jax.custom_batching.custom_vmap
    def execute_primal(model: DynamicsModel, q: Array, qd: Array) -> DynamicsTerms:
        """Execute a scalar request through a temporary one-item batch.

        Args:
            model: System model supplied to the family executor.
            q: Generalized coordinates for one environment.
            qd: Generalized velocities for the same environment.

        Returns:
            Scalar inertia, Coriolis/centrifugal, and gravity terms.
        """

        batched = execute_batch(model, q[None, :], qd[None, :])
        return jax.tree.map(lambda value: value[0], batched)

    @execute_primal.def_vmap
    def vmap_rule(
        axis_size: int,
        in_batched: tuple[Any, bool, bool],
        model: DynamicsModel,
        q: Array,
        qd: Array,
    ) -> tuple[DynamicsTerms, tuple[bool, bool, bool]]:
        """Combine mapped environments into one family executor call.

        Args:
            axis_size: Size of the mapped axis.
            in_batched: Boolean PyTree indicating which inputs carry that axis.
            model: Unbatched system model.
            q: Scalar or mapped generalized coordinates.
            qd: Scalar or mapped generalized velocities.

        Returns:
            Batched dynamics terms and flags marking every output as batched.

        Raises:
            ValueError: If a caller attempts to batch model parameters.
        """

        model_batched, q_batched, qd_batched = in_batched
        if any(jax.tree.leaves(model_batched)):
            raise ValueError(
                f"Batching over {family_name} model parameters is not supported."
            )
        if not q_batched:
            q = jnp.broadcast_to(q, (axis_size, *q.shape))
        if not qd_batched:
            qd = jnp.broadcast_to(qd, (axis_size, *qd.shape))
        return execute_batch(model, q, qd), (True, True, True)

    @eqx.filter_custom_jvp
    def evaluate_terms(model: DynamicsModel, q: Array, qd: Array) -> DynamicsTerms:
        """Evaluate scalar forward-only dynamics terms.

        Args:
            model: System model supplied to the family executor.
            q: Generalized coordinates for one environment.
            qd: Generalized velocities for the same environment.

        Returns:
            Scalar inertia, Coriolis/centrifugal, and gravity terms.
        """

        return execute_primal(model, q, qd)

    @evaluate_terms.def_jvp
    def jvp_rule(
        primals: tuple[DynamicsModel, Array, Array],
        tangents: tuple[Any, Array | None, Array | None],
    ) -> tuple[DynamicsTerms, DynamicsTerms]:
        """Differentiate the model's JAX dynamics assembly.

        Args:
            primals: Model, generalized coordinates, and velocities.
            tangents: Equinox-filtered tangents corresponding to ``primals``.

        Returns:
            JAX-backed primal terms and their directional derivatives.
        """

        return eqx.filter_jvp(
            lambda model, q, qd: model._assemble_dynamics_terms(q, qd),
            primals,
            tangents,
        )

    return evaluate_terms


@eqx.filter_custom_jvp
def evaluate_forward_dynamics(
    model: ForwardDynamicsModel,
    t: Array,
    y: Array,
    actuation_args: tuple | None,
    backend: ExecutionBackend | None,
) -> Array:
    """Evaluate compiled forward dynamics with transform-aware backend routing.

    This boundary does not implement a new analytical derivative. Its custom
    JVP exists solely to prevent a compiled primal calculation from staging a
    forward-only Warp callback before JAX selects a derivative rule. Primal
    calls use the model's configured backend; forward- and reverse-mode
    transformations evaluate the same forward-dynamics equations with JAX term
    assembly.

    Args:
        model: System implementing the forward-dynamics execution contract.
        t: Current integration time.
        y: State vector accepted by the system's forward dynamics.
        actuation_args: Optional actuation and external-force arguments forwarded
            unchanged to the system.
        backend: Optional per-call backend override. ``None`` uses the model's
            configured backend.

    Returns:
        The state derivative produced by the model's compiled forward-dynamics
        implementation.
    """

    return _assemble_forward_dynamics(model, t, y, actuation_args, backend=backend)


@evaluate_forward_dynamics.def_jvp
def _evaluate_forward_dynamics_jvp(
    primals: tuple[
        ForwardDynamicsModel,
        Array,
        Array,
        tuple | None,
        ExecutionBackend | None,
    ],
    tangents: tuple[Any, Any, Any, Any, Any],
) -> tuple[Array, Array]:
    """Differentiate forward dynamics using the model's JAX term assembly.

    Args:
        primals: Model, time, state, optional actuation arguments, and backend
            supplied to :func:`evaluate_forward_dynamics`.
        tangents: Tangents corresponding to ``primals`` as filtered by Equinox.

    Returns:
        A pair containing the JAX-backed primal state derivative and its
        directional derivative.
    """

    model, t, y, actuation_args, _backend = primals
    model_tangent, t_tangent, y_tangent, actuation_tangent, _backend_tangent = tangents
    return eqx.filter_jvp(
        lambda model_, t_, y_, actuation_args_: _assemble_forward_dynamics(
            model_, t_, y_, actuation_args_, backend="jax"
        ),
        (model, t, y, actuation_args),
        (model_tangent, t_tangent, y_tangent, actuation_tangent),
    )


@eqx.filter_jit
def _assemble_forward_dynamics(
    model: ForwardDynamicsModel,
    t: Array,
    y: Array,
    actuation_args: tuple | None,
    *,
    backend: ExecutionBackend | None,
) -> Array:
    """Assemble generalized forces and solve one forward-dynamics request.

    This shared implementation keeps the public system methods small and makes
    the force convention identical for GVS, PCS, and PlanarPCS. The selected
    backend affects only ``(B, Cqd, G)`` assembly; passive forces, actuation,
    damping, and the inertia solve remain JAX operations.

    Args:
        model: System implementing the forward-dynamics execution contract.
        t: Current integration time. The supported autonomous systems accept it
            for solver compatibility and do not otherwise use it.
        y: State vector ``[q, qd]`` for one environment.
        actuation_args: Optional tuple ``(u,)`` or ``(u, tau_ext)``. Missing
            actuation and external force values default to zero.
        backend: Backend used to assemble dynamics terms. ``None`` uses the
            model's configured backend.

    Returns:
        State time derivative ``[qd, qdd]`` with the same shape as ``y``.

    Raises:
        ValueError: If ``actuation_args`` has a length other than one or two.
    """

    del t
    q, qd = jnp.split(y, 2)
    if actuation_args is None:
        u, tau_ext = None, None
    elif len(actuation_args) == 1:
        u = actuation_args[0]
        tau_ext = None
    elif len(actuation_args) == 2:
        u, tau_ext = actuation_args
    else:
        raise ValueError("actuation_args must be a tuple of length 1 or 2.")

    if u is None:
        u = jnp.zeros((model.num_actuators,))
    if tau_ext is None:
        tau_ext = jnp.zeros((q.shape[-1],))

    inertia, coriolis_qd, gravity = model.dynamics_terms(q, qd, backend=backend)
    elastic = model.elastic_force(q)
    actuation = model.actuation_force(q, u, qd=qd)
    rhs = (
        actuation
        + tau_ext
        - coriolis_qd
        - gravity
        - elastic
        - model.damping_matrix(q) @ qd
    )
    qdd = model._solve_inertia(inertia, rhs)
    return jnp.concatenate([qd, qdd])


__all__ = [
    "BatchExecutor",
    "KinematicsBatchExecutor",
    "evaluate_forward_dynamics",
    "make_dynamics_evaluator",
    "make_kinematics_evaluators",
]
