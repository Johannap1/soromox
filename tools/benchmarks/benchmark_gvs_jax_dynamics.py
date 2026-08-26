#!/usr/bin/env python3

"""Benchmark JAX GVS dynamics over coupled model and batch-size sweeps.

The CLI deliberately imports JAX only after processing ``--device`` so CPU and
GPU runs cannot silently use different backends than requested. CPU runs can
also require an exact one-core affinity mask, making accidental E-core or
multi-core measurements fail loudly instead of contaminating the results.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _configure_device_before_jax_import(argv: Sequence[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    args, _ = parser.parse_known_args(argv)
    if args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif args.device == "gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"
    return str(args.device)


_EARLY_DEVICE = _configure_device_before_jax_import(sys.argv[1:])

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from soromox.systems import (  # noqa: E402
    GVS,
    GVSSegment,
    JointSpec,
    LinkSpec,
    StrainBasisSpec,
)
from tools.benchmarks._benchmark_common import block_until_ready  # noqa: E402

jax.config.update("jax_enable_x64", True)

Array = jax.Array
Tree = Any


def _variant_options(
    system: GVS, variant: str
) -> tuple[bool, bool, bool, int, str, str]:
    """Return velocity, active-prefix, compact-basis, and bucket switches."""
    if variant == "baseline":
        return False, False, False, 0, "uniform", "none"
    if variant == "velocity":
        return True, False, False, 0, "uniform", "none"
    if variant in ("fixed_compact", "fixed_optimized"):
        return True, False, True, 0, "uniform", "none"
    for name, local_policy in (
        ("local_dofs_", "dofs"),
        ("local_full_", "full"),
    ):
        if variant.startswith(name):
            return (
                True,
                False,
                True,
                int(variant.rsplit("_", maxsplit=1)[1]),
                "uniform",
                local_policy,
            )
    for name, policy in (
        ("bucket_uniform_", "uniform"),
        ("bucket_equal_", "equal_count"),
        ("bucket_optimal_", "cost_optimal"),
    ):
        if variant.startswith(name):
            return (
                True,
                False,
                True,
                int(variant.rsplit("_", maxsplit=1)[1]),
                policy,
                "none",
            )
    if variant == "active_prefix":
        return True, True, False, 0, "uniform", "none"
    if variant == "compact_basis":
        return True, True, True, 0, "uniform", "none"
    if variant == "optimized":
        use_optimization = system.num_segments > 1
        return (
            use_optimization,
            use_optimization,
            use_optimization,
            0,
            "uniform",
            "none",
        )
    raise ValueError(f"Unknown dynamics variant: {variant}")


def _parse_strain_gauss_pair(value: str) -> tuple[int, int]:
    try:
        order_text, gauss_text = value.split(":", maxsplit=1)
        order, gauss_points = int(order_text), int(gauss_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "expected ORDER:GAUSS_POINTS, for example 5:9"
        ) from error
    if order < 0:
        raise argparse.ArgumentTypeError("strain order must be non-negative")
    if gauss_points < 5:
        raise argparse.ArgumentTypeError("GVS requires at least five Gauss points")
    return order, gauss_points


def _active_cpu_affinity() -> tuple[int, ...] | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    return tuple(sorted(os.sched_getaffinity(0)))


def _build_system(
    num_segments: int,
    strain_order: int,
    gauss_points: int,
    joint_type: str,
    topology: str = "homogeneous",
) -> GVS:
    def build_segment(order: int, segment_gauss_points: int) -> GVSSegment:
        return GVSSegment(
            link=LinkSpec.circular(
                young_modulus=1.0e6,
                shear_modulus=1.0e6 / 2.9,
                density=980.0,
                material_damping_coefficient=2.5e3,
                length=0.25,
                radius=0.02,
                reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ),
            joint=JointSpec(type=joint_type, axis="z"),
            basis=StrainBasisSpec(
                type="legendre",
                strain_selector=[1, 1, 1, 1, 1, 1],
                basis_order=order,
            ),
            num_gauss_points=segment_gauss_points,
        )

    high_shape = (strain_order, gauss_points)
    low_shape = (max(0, strain_order - 2), max(5, gauss_points - 2))
    if topology == "homogeneous":
        shapes = [high_shape] * num_segments
    elif topology == "alternating":
        shapes = [
            low_shape if index % 2 == 0 else high_shape for index in range(num_segments)
        ]
    elif topology == "grouped":
        split = (num_segments + 1) // 2
        shapes = [low_shape] * split + [high_shape] * (num_segments - split)
    else:
        raise ValueError(f"unknown segment topology: {topology}")
    return GVS.from_segments(
        segments=[build_segment(*shape) for shape in shapes],
        gravity=jnp.array([0.0, 0.0, 9.81], dtype=jnp.float64),
    )


def _batched_inputs(system: GVS, batch_size: int) -> tuple[Array, Array]:
    columns = jnp.arange(system.num_dofs, dtype=jnp.float64)[None, :]
    environments = jnp.arange(batch_size, dtype=jnp.float64)[:, None]
    phase = 0.17 * columns + 0.013 * environments
    q = 0.08 * jnp.sin(phase) + 0.015 * jnp.cos(0.31 * phase)
    qd = 0.12 * jnp.cos(0.73 * phase) - 0.02 * jnp.sin(0.19 * phase)
    return q, qd


def _dynamics_callable(
    system: GVS, batch_size: int, variant: str
) -> Callable[[Array, Array], Tree]:
    (
        reuse_recurrence_velocity,
        use_active_prefix,
        use_compact_basis,
        reduction_bucket_count,
        reduction_bucket_policy,
        local_shape_policy,
    ) = _variant_options(system, variant)

    def single(q: Array, qd: Array) -> Tree:
        return system._dynamics_terms_impl(
            q,
            qd,
            reuse_recurrence_velocity=reuse_recurrence_velocity,
            use_active_prefix=use_active_prefix,
            use_compact_basis=use_compact_basis,
            reduction_bucket_count=reduction_bucket_count,
            reduction_bucket_policy=reduction_bucket_policy,
            local_shape_policy=local_shape_policy,
        )

    if batch_size == 1 and jax.default_backend() == "cpu":
        return lambda q, qd: single(q[0], qd[0])
    return lambda q, qd: jax.vmap(single)(q, qd)


def _forward_dynamics_callable(
    system: GVS, batch_size: int, variant: str
) -> Callable[[Array, Array], Tree]:
    (
        reuse_recurrence_velocity,
        use_active_prefix,
        use_compact_basis,
        reduction_bucket_count,
        reduction_bucket_policy,
        local_shape_policy,
    ) = _variant_options(system, variant)

    def single(q: Array, qd: Array) -> Array:
        inertia, coriolis_qd, gravity = system._dynamics_terms_impl(
            q,
            qd,
            reuse_recurrence_velocity=reuse_recurrence_velocity,
            use_active_prefix=use_active_prefix,
            use_compact_basis=use_compact_basis,
            reduction_bucket_count=reduction_bucket_count,
            reduction_bucket_policy=reduction_bucket_policy,
            local_shape_policy=local_shape_policy,
        )
        u = jnp.zeros((system.num_actuators,), dtype=q.dtype)
        actuation = system.actuation_force(q, u, qd=qd)
        rhs = (
            actuation
            - coriolis_qd
            - gravity
            - system.elastic_force(q)
            - system.damping_matrix(q) @ qd
        )
        qdd = system._solve_inertia(inertia, rhs)
        return jnp.concatenate([qd, qdd])

    if batch_size == 1 and jax.default_backend() == "cpu":
        return lambda q, qd: single(q[0], qd[0])
    return jax.vmap(single)


def _measure(
    fn: Callable[[Array, Array], Tree],
    q: Array,
    qd: Array,
    *,
    repeats: int,
    warmup_iterations: int,
) -> dict[str, Any]:
    jitted = jax.jit(fn)

    start = time.perf_counter()
    lowered = jitted.lower(q, qd)
    lower_time = time.perf_counter() - start
    stablehlo_text = lowered.as_text()

    start = time.perf_counter()
    executable = lowered.compile()
    compile_time = time.perf_counter() - start

    start = time.perf_counter()
    first_result = executable(q, qd)
    block_until_ready(first_result)
    first_execution_time = time.perf_counter() - start

    for _ in range(warmup_iterations):
        block_until_ready(executable(q, qd))

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        block_until_ready(executable(q, qd))
        samples.append(time.perf_counter() - start)

    ordered = sorted(samples)
    p90_index = min(len(ordered) - 1, int(0.9 * len(ordered)))
    return {
        "lower_time_s": lower_time,
        "compile_time_s": compile_time,
        "stablehlo_text_bytes": len(stablehlo_text.encode("utf-8")),
        "stablehlo_operation_count": stablehlo_text.count("stablehlo."),
        "stablehlo_case_count": stablehlo_text.count("stablehlo.case"),
        "stablehlo_while_count": stablehlo_text.count("stablehlo.while"),
        "first_execution_time_s": first_execution_time,
        "execution_median_s": statistics.median(samples),
        "execution_min_s": min(samples),
        "execution_p90_s": ordered[p90_index],
        "execution_samples_s": samples,
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--segment-counts", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument(
        "--strain-gauss-pairs",
        nargs="+",
        type=_parse_strain_gauss_pair,
        default=[(0, 5), (1, 5), (3, 7), (5, 9)],
        metavar="ORDER:GAUSS_POINTS",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--joint-type", choices=("fixed", "revolute"), default="revolute"
    )
    parser.add_argument(
        "--topology",
        choices=("homogeneous", "alternating", "grouped"),
        default="homogeneous",
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=("dynamics_terms", "forward_dynamics"),
        default=["dynamics_terms"],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=(
            "baseline",
            "velocity",
            "fixed_compact",
            "fixed_optimized",
            "bucket_uniform_2",
            "bucket_uniform_4",
            "bucket_uniform_8",
            "bucket_equal_4",
            "bucket_equal_8",
            "bucket_optimal_4",
            "bucket_optimal_8",
            "local_dofs_4",
            "local_dofs_8",
            "local_full_4",
            "local_full_8",
            "active_prefix",
            "compact_basis",
            "optimized",
        ),
        default=["velocity"],
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument(
        "--require-cpu-core",
        type=int,
        help="Fail unless the CPU affinity is exactly this one logical core.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(value < 1 for value in args.segment_counts):
        parser.error("--segment-counts values must be positive")
    if any(value < 1 for value in args.batch_sizes):
        parser.error("--batch-sizes values must be positive")
    if args.device == "cpu" and args.batch_sizes != [1]:
        parser.error("CPU evaluation is intentionally restricted to batch size 1")
    if args.repeats < 1 or args.warmup_iterations < 0:
        parser.error("--repeats must be positive and warmup iterations non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if _EARLY_DEVICE != "auto" and args.device != _EARLY_DEVICE:
        raise RuntimeError("--device changed after JAX was imported")
    if jax.default_backend() != args.device:
        raise RuntimeError(
            f"requested {args.device}, but JAX selected {jax.default_backend()}"
        )

    affinity = _active_cpu_affinity()
    if args.require_cpu_core is not None and affinity != (args.require_cpu_core,):
        raise RuntimeError(
            f"expected CPU affinity ({args.require_cpu_core},), got {affinity}"
        )

    metadata = {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "cpu_affinity": affinity,
        "joint_type": args.joint_type,
        "topology": args.topology,
        "repeats": args.repeats,
        "warmup_iterations": args.warmup_iterations,
    }
    print(json.dumps(metadata, indent=2))

    results = []
    for num_segments in args.segment_counts:
        for strain_order, gauss_points in args.strain_gauss_pairs:
            system = _build_system(
                num_segments,
                strain_order,
                gauss_points,
                args.joint_type,
                args.topology,
            )
            for batch_size in args.batch_sizes:
                q, qd = _batched_inputs(system, batch_size)
                for operation in args.operations:
                    for variant in args.variants:
                        fn = (
                            _dynamics_callable(system, batch_size, variant)
                            if operation == "dynamics_terms"
                            else _forward_dynamics_callable(system, batch_size, variant)
                        )
                        print(
                            f"segments={num_segments} order={strain_order} "
                            f"gauss={gauss_points} batch={batch_size} "
                            f"dofs={system.num_dofs} operation={operation} "
                            f"variant={variant}",
                            flush=True,
                        )
                        timing = _measure(
                            fn,
                            q,
                            qd,
                            repeats=args.repeats,
                            warmup_iterations=args.warmup_iterations,
                        )
                        row = {
                            "num_segments": num_segments,
                            "strain_order": strain_order,
                            "gauss_points": gauss_points,
                            "batch_size": batch_size,
                            "num_dofs": system.num_dofs,
                            "operation": operation,
                            "topology": args.topology,
                            "variant": variant,
                            **timing,
                        }
                        results.append(row)
                        print(
                            f"  median={1e3 * timing['execution_median_s']:.3f} ms "
                            f"p90={1e3 * timing['execution_p90_s']:.3f} ms",
                            flush=True,
                        )

    payload = {"metadata": metadata, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
