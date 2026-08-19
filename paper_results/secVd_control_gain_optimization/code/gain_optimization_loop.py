"""Shared multi-start gain-optimization loop for the Section Vd study.

Both Section Vd generators run the same Optax loop over closed-loop rollouts, so
it lives here and each generator supplies only its own loss and optimizer.

The loop optimizes ``B`` independent starts at once. Everything is vectorized
here rather than in the callers, so a caller cannot batch the evaluation while
leaving the optimizer unbatched: the gradient function, the optimizer's ``init``
and its ``update`` are all wrapped in the same ``vmap``. Because the optimizer
runs inside the ``vmap``, ``optax.multi_transform`` label trees keep matching the
unbatched parameter tree, and a global-norm clip clips each start on its own --
which is what independent starts require.

The loop records the parameters each loss was measured at, not the ones the
subsequent update produced, so ``argmin`` over the losses indexes the matching
gains. No unused update is computed after the final requested evaluation.

A start whose loss, trajectory, gradient or optimizer update turns non-finite is
frozen at its current parameters and marked invalid in ``finite_mask`` from that
iteration on; the remaining starts continue. Freezing at the *current* rather
than the last-good parameters is deliberate: re-evaluating the last good ones
would produce a finite loss again and resurrect a start that has already failed.
Checking the gradient matters as much as the trajectory, since a forward rollout
can stay healthy while the reverse pass produces ``NaN``.

Per-iteration rollouts are never accumulated. At 100 iterations and six starts
they would run to gigabytes, so the loop keeps each start's iteration-zero
rollout and a running per-start best, which is all the archive needs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
from jax import Array
from jax import numpy as jnp

from soromox.utils.diagnostics import format_nonfinite_report, nonfinite_leaves

__all__ = ["OptimizationHistory", "run_gain_optimization"]


def _broadcast_mask(mask: Array, ndim: int) -> Array:
    """Shape a per-start boolean mask for ``jnp.where`` against an ``ndim`` leaf."""
    return mask.reshape(mask.shape + (1,) * (ndim - 1))


def _select(mask: Array, chosen: Any, fallback: Any) -> Any:
    """Take ``chosen`` for starts where ``mask`` holds, ``fallback`` elsewhere."""
    return jax.tree_util.tree_map(
        lambda new, old: jnp.where(_broadcast_mask(mask, jnp.ndim(new)), new, old),
        chosen,
        fallback,
    )


def _finite_per_start(tree: Any, batch_size: int) -> Array:
    """Reduce every batched leaf of ``tree`` to one finiteness flag per start.

    Args:
        tree: Any pytree whose batched leaves lead with the start axis.
        batch_size: Expected width of that axis.

    Returns:
        Boolean array of shape ``(batch_size,)``.
    """
    alive = jnp.ones((batch_size,), dtype=bool)
    for leaf in jax.tree_util.tree_leaves(tree):
        array = jnp.asarray(leaf)
        # Optimizer states carry unbatched bookkeeping leaves; they cannot be
        # attributed to a start and are covered by the batched leaves anyway.
        if array.ndim == 0 or array.shape[0] != batch_size:
            continue
        if array.dtype.kind not in "fc":
            continue
        alive &= jnp.all(jnp.isfinite(array.reshape(batch_size, -1)), axis=1)
    return alive


def _slice_start(tree: Any, index: int, batch_size: int) -> Any:
    """Extract one start's slice of every batched leaf, for reporting."""
    return jax.tree_util.tree_map(
        lambda leaf: (
            jnp.asarray(leaf)[index]
            if jnp.ndim(leaf) > 0 and jnp.shape(leaf)[0] == batch_size
            else leaf
        ),
        tree,
    )


def _report_dead_starts(
    dead: list[int], candidates: dict[str, Any], iteration: int, batch_size: int
) -> str:
    """Name the leaves that went non-finite, for the first start that failed.

    Args:
        dead: Indices of starts that failed at this iteration.
        candidates: Trees inspected this iteration, keyed by stage name.
        iteration: Iteration index, used in the message.
        batch_size: Width of the start axis.

    Returns:
        A single-line summary naming the offending leaves.
    """
    index = dead[0]
    for stage, tree in candidates.items():
        report = nonfinite_leaves(_slice_start(tree, index, batch_size))
        if report:
            return format_nonfinite_report(
                f"iteration {iteration}, start {index} {stage}", report
            )
    return f"iteration {iteration}, start {index}: no non-finite leaf located"


@dataclass
class OptimizationHistory:
    """Per-iteration record of a multi-start gain-optimization run.

    Every list is indexed by completed iteration and all lists have the same
    length. Entry ``k`` of :attr:`opt_vars` holds the parameters that produced
    entry ``k`` of :attr:`loss`, per start.

    Attributes:
        loss: Losses of shape ``(B,)`` measured at each recorded iterate. Entries
            for starts marked invalid in :attr:`finite_mask` may be non-finite.
        opt_vars: Optimization variables the corresponding losses were measured
            at, batched over starts.
        finite_mask: Per-iteration validity flags of shape ``(B,)``.
        time_iter: Wall-clock duration of each iteration in seconds. The first
            entry includes tracing and compilation.
        init_aux: Iteration-zero rollout outputs, batched over starts.
        best_aux: Each start's own lowest-loss rollout outputs.
        best_loss: Each start's own lowest loss, shape ``(B,)``.
        best_iteration: Each start's own lowest-loss iteration, shape ``(B,)``.
        stopped_early: Whether the loop halted before ``num_iters``.
        stop_reason: Human-readable description of the early stop, or ``None``.
    """

    batch_size: int = 1
    loss: list[Array] = field(default_factory=list)
    opt_vars: list[Any] = field(default_factory=list)
    finite_mask: list[Array] = field(default_factory=list)
    time_iter: list[float] = field(default_factory=list)
    init_aux: dict[str, Array] | None = None
    best_aux: dict[str, Array] | None = None
    best_loss: Array | None = None
    best_iteration: Array | None = None
    stopped_early: bool = False
    stop_reason: str | None = None

    def __len__(self) -> int:
        """Return the number of recorded iterations."""
        return len(self.loss)

    def _require_recorded(self, action: str) -> None:
        """Raise if no iterate was recorded, reporting why the loop stopped.

        Args:
            action: What the caller was trying to do, used in the message.

        Raises:
            RuntimeError: If no iterate was recorded.
        """
        if not self.loss:
            raise RuntimeError(
                f"No optimization iterate was recorded; nothing to {action}. "
                f"Stop reason: {self.stop_reason}"
            )

    def loss_history(self) -> Array:
        """Return the losses as an ``(iterations, B)`` array."""
        self._require_recorded("stack")
        return jnp.stack(self.loss, axis=0)

    def mask_history(self) -> Array:
        """Return the validity flags as an ``(iterations, B)`` array."""
        self._require_recorded("stack")
        return jnp.stack(self.finite_mask, axis=0)

    def best_batch(self) -> int:
        """Return the start that attained the lowest loss overall.

        Returns:
            Index into the start axis.

        Raises:
            RuntimeError: If no iterate was recorded.
        """
        self._require_recorded("select")
        return int(
            jnp.argmin(jnp.where(jnp.isfinite(self.best_loss), self.best_loss, jnp.inf))
        )

    def dead_starts(self) -> list[int]:
        """Return the starts that never produced a finite iterate."""
        self._require_recorded("inspect")
        return jnp.flatnonzero(~self.mask_history().any(axis=0)).tolist()


def run_gain_optimization(
    gradient_fn: Callable[[Any], tuple[tuple[Array, dict[str, Array]], Any]],
    optimizer,
    opt_vars: Any,
    num_iters: int,
    *,
    batch_size: int,
    progress_label: str = "Optimization",
) -> OptimizationHistory:
    """Run the Optax gain-optimization loop over ``batch_size`` independent starts.

    Args:
        gradient_fn: Callable mapping **one start's** optimization variables to
            ``((loss, aux), gradient)``, typically
            ``value_and_grad(evaluate_closed_loop_system, has_aux=True)`` with the
            controller already bound. The loop vectorizes and jits it.
        optimizer: Optax optimizer built for the unbatched parameter tree.
        opt_vars: Initial optimization variables whose leaves lead with a start
            axis of width ``batch_size``.
        num_iters: Number of iterations to attempt. Must be at least one.
        batch_size: Number of independent starts.
        progress_label: Prefix used in the progress log line.

    Returns:
        :class:`OptimizationHistory` whose entries are pairwise consistent.

    Raises:
        ValueError: If ``num_iters`` or ``batch_size`` is smaller than one.
    """
    if num_iters < 1:
        raise ValueError("num_iters must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    import optax  # Imported lazily; only the paper_results extra provides it.

    batched_gradient = jax.jit(jax.vmap(gradient_fn))
    batched_init = jax.jit(jax.vmap(optimizer.init))

    @jax.jit
    @jax.vmap
    def batched_step(grad, state, params):
        updates, next_state = optimizer.update(grad, state, params=params)
        return optax.apply_updates(params, updates), next_state, updates

    opt_state = batched_init(opt_vars)
    history = OptimizationHistory(batch_size=batch_size)
    active = jnp.ones((batch_size,), dtype=bool)
    best_loss = jnp.full((batch_size,), jnp.inf)
    best_iteration = jnp.zeros((batch_size,), dtype=jnp.int32)
    best_aux: dict[str, Array] | None = None

    for iteration in range(num_iters):
        iter_start = time.time()

        # Keep the parameters the loss belongs to before the update moves them.
        opt_vars_evaluated = opt_vars
        try:
            (loss, aux), grad = batched_gradient(opt_vars)
            jax.block_until_ready((loss, aux, grad))
        except KeyboardInterrupt:
            history.stopped_early = True
            history.stop_reason = (
                f"interrupted during iteration {iteration + 1}; preserving "
                f"{len(history)} completed iterations"
            )
            print(f"\n[WARNING] {history.stop_reason}.")
            break

        evaluation = {
            "loss": loss,
            "aux": aux,
            "gradient": grad,
            "opt_vars": opt_vars_evaluated,
        }
        inspected: dict[str, Any] = {"evaluation": evaluation}
        evaluated_ok = active & _finite_per_start(evaluation, batch_size)

        if iteration == 0:
            history.init_aux = aux
            best_aux = aux

        improved = evaluated_ok & (loss < best_loss)
        best_loss = jnp.where(improved, loss, best_loss)
        best_iteration = jnp.where(improved, iteration, best_iteration)
        best_aux = _select(improved, aux, best_aux)

        # Do not compute an update after the final requested evaluation. A start
        # whose update is non-finite keeps this finite evaluation and freezes.
        if iteration < num_iters - 1:
            opt_vars_next, opt_state_next, updates = batched_step(
                grad, opt_state, opt_vars_evaluated
            )
            jax.block_until_ready((opt_vars_next, opt_state_next, updates))
            update = {
                "updates": updates,
                "opt_state_next": opt_state_next,
                "opt_vars_next": opt_vars_next,
            }
            inspected["update"] = update
            update_ok = _finite_per_start(update, batch_size)
            advance = evaluated_ok & update_ok
            opt_vars = _select(advance, opt_vars_next, opt_vars_evaluated)
            opt_state = _select(advance, opt_state_next, opt_state)
        else:
            advance = evaluated_ok

        iter_duration = time.time() - iter_start
        history.loss.append(loss)
        history.opt_vars.append(opt_vars_evaluated)
        history.finite_mask.append(evaluated_ok)
        history.time_iter.append(iter_duration)

        newly_dead = jnp.flatnonzero(active & ~advance).tolist()
        if newly_dead:
            detail = _report_dead_starts(newly_dead, inspected, iteration, batch_size)
            print(
                f"\n[WARNING] Starts {newly_dead} became non-finite at iteration "
                f"{iteration}; freezing them and continuing. {detail}"
            )
        active = advance

        if not bool(jnp.any(active)):
            history.stopped_early = True
            history.stop_reason = f"every start was non-finite by iteration {iteration}"
            print(f"\n[WARNING] {history.stop_reason}. Stopping.")
            break

        if iteration == 0:
            print(
                f"\n[INFO] Compilation + first execution time = {iter_duration:.2f} s"
            )
            continue

        steady_times = history.time_iter[1:]
        t_left = (num_iters - iteration - 1) * sum(steady_times) / len(steady_times)
        eta = time.localtime(time.time() + t_left)
        print(
            f"{progress_label}: {100 * (iteration + 1) / num_iters:3.1f} %  |  "
            f"iteration {iteration + 1:>4d} of {num_iters:<4d}  |  "
            f"best loss = {float(jnp.min(best_loss)):.4g}  |  "
            f"{int(jnp.sum(active))}/{batch_size} starts alive  |  "
            f"iter time = {iter_duration:>.2f} s  |  "
            f"ETA = {eta.tm_mday:02d}/{eta.tm_mon:02d}/{eta.tm_year} "
            f"{eta.tm_hour:02d}:{eta.tm_min:02d}",
            end="\r",
        )

    history.best_aux = best_aux
    history.best_loss = best_loss
    history.best_iteration = best_iteration
    print(
        f"\n{progress_label}: recorded {len(history)} of {num_iters} iterations"
        + ("  (stopped early)" if history.stopped_early else "")
    )
    return history
