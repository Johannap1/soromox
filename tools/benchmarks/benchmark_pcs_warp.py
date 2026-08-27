#!/usr/bin/env python3

"""Compare five-point PlanarPCS/PCS JAX and Warp dynamics execution."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any


def _configure_device() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    args, _ = parser.parse_known_args()
    os.environ["JAX_PLATFORMS"] = "cpu" if args.device == "cpu" else "cuda"


_configure_device()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from soromox.systems import SystemState  # noqa: E402
from tools.benchmarks._benchmark_common import (  # noqa: E402
    _pcs_context,
    _pcs_factory,
    _planar_pcs_context,
    _planar_pcs_factory,
    block_until_ready,
)

jax.config.update("jax_enable_x64", True)

GAUSS_POINTS = 5


def _terms(model: Any, q: jax.Array, qd: jax.Array, backend: str):
    if backend == "jax" and not hasattr(model, "backend"):
        if q.ndim == 1:
            return model.dynamics_terms(q, qd)
        return jax.vmap(model.dynamics_terms)(q, qd)
    return model.dynamics_terms(q, qd, backend=backend)


def _forward(model: Any, t: jax.Array, y: jax.Array, args: tuple[Any, ...]):
    if y.ndim == 1:
        return model.forward_dynamics(t, y, args)
    return jax.vmap(model.forward_dynamics, in_axes=(None, 0, None))(t, y, args)


def _rollout(
    model: Any,
    y: jax.Array,
    u: jax.Array,
    tau: jax.Array,
    duration: float,
    solver_dt: float,
    save_dt: float,
):
    def one(initial_y: jax.Array):
        return model.rollout_to(
            initial_state=SystemState(t=0.0, y=initial_y),
            u=u,
            tau_ext=tau,
            t1=duration,
            solver_dt=solver_dt,
            save_dt=save_dt,
        )

    if y.ndim == 1:
        return one(y)
    return jax.vmap(one)(y)


def _measure(fn, repeats: int) -> dict[str, float]:
    # Keep first-call compilation measurements independent across operations.
    # This deliberately retains Warp's on-disk module cache, which is reported
    # separately from the JAX wrapper compilation in the production evaluation.
    jax.clear_caches()
    compiled = jax.jit(fn)
    start = time.perf_counter()
    first = compiled()
    block_until_ready(first)
    first_s = time.perf_counter() - start

    for _ in range(20):
        block_until_ready(compiled())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        block_until_ready(compiled())
        samples.append(time.perf_counter() - start)
    warm_s = statistics.median(samples)
    return {
        "first_call_s": first_s,
        "compile_estimate_s": max(0.0, first_s - warm_s),
        "warm_s": warm_s,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=("planar_pcs", "pcs"),
        default=("planar_pcs", "pcs"),
    )
    parser.add_argument("--segments", nargs="+", type=int, default=(1, 4, 8, 16))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=(1, 64, 256, 1024))
    parser.add_argument("--backends", nargs="+", choices=("jax", "warp"), default=("jax",))
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=("terms", "forward", "rollout"),
        default=("terms",),
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--duration", type=float, default=0.001)
    parser.add_argument("--solver-dt", type=float, default=0.00001)
    parser.add_argument("--save-dt", type=float, default=0.001)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if any(value < 1 for value in (*args.segments, *args.batch_sizes)):
        raise ValueError("segment counts and batch sizes must be positive")
    if jax.default_backend() != args.device:
        raise RuntimeError(
            f"requested {args.device!r}, selected {jax.default_backend()!r}"
        )

    factories = {
        "planar_pcs": (_planar_pcs_factory, _planar_pcs_context),
        "pcs": (_pcs_factory, _pcs_context),
    }
    results: list[dict[str, Any]] = []
    for system_name in args.systems:
        factory, context_factory = factories[system_name]
        for segments in args.segments:
            model = factory(segments, GAUSS_POINTS)
            context = context_factory(model)
            q0 = context["q"]
            qd0 = context["qd"]
            y0 = context["y"]
            for batch_size in args.batch_sizes:
                q = jnp.broadcast_to(q0, (batch_size, *q0.shape))
                qd = jnp.broadcast_to(qd0, (batch_size, *qd0.shape))
                y = jnp.broadcast_to(y0, (batch_size, *y0.shape))
                if batch_size == 1:
                    q, qd, y = q[0], qd[0], y[0]
                for backend in args.backends:
                    execution_model = factory(
                        segments, GAUSS_POINTS, backend=backend
                    )
                    for operation in args.operations:
                        if operation == "terms":
                            def fn(
                                m=execution_model, q_=q, qd_=qd, b=backend
                            ):
                                return _terms(m, q_, qd_, b)

                        elif operation == "forward":
                            def fn(m=execution_model, y_=y, c=context):
                                return _forward(
                                    m, c["t"], y_, (c["u"], c["tau_ext"])
                                )

                        else:
                            def fn(m=execution_model, y_=y, c=context):
                                return _rollout(
                                    m,
                                    y_,
                                    c["u"],
                                    c["tau_ext"],
                                    args.duration,
                                    args.solver_dt,
                                    args.save_dt,
                                )
                        print(
                            f"{system_name} s={segments} b={batch_size} "
                            f"backend={backend} operation={operation}",
                            flush=True,
                        )
                        timing = _measure(fn, args.repeats)
                        result = {
                            "system": system_name,
                            "segments": segments,
                            "gauss_points": GAUSS_POINTS,
                            "num_dofs": model.num_dofs,
                            "batch_size": batch_size,
                            "device": args.device,
                            "backend": backend,
                            "operation": operation,
                            **timing,
                        }
                        results.append(result)
                        print(result, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
