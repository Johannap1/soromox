#!/usr/bin/env python3

"""Run each GVS JAX compilation benchmark cell in a fresh process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parse_pair(value: str) -> tuple[int, int]:
    order, gauss_points = value.split(":", maxsplit=1)
    return int(order), int(gauss_points)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--segment-counts", nargs="+", type=int, required=True)
    parser.add_argument(
        "--strain-gauss-pairs", nargs="+", type=_parse_pair, required=True
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument(
        "--topology",
        choices=("homogeneous", "alternating", "grouped"),
        default="homogeneous",
    )
    parser.add_argument("--cpu-core", type=int)
    parser.add_argument("--cell-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cpu" and args.cpu_core is None:
        parser.error("--cpu-core is required for CPU measurements")
    if args.device == "cpu" and args.batch_sizes != [1]:
        parser.error("CPU compilation measurements require batch size 1")
    return args


def main() -> int:
    args = parse_args()
    benchmark = Path(__file__).with_name("benchmark_gvs_jax_dynamics.py")
    args.cell_directory.mkdir(parents=True, exist_ok=True)
    aggregate: dict[str, Any] = {
        "method": "fresh Python/JAX process per result",
        "results": [],
    }

    for segments in args.segment_counts:
        for order, gauss_points in args.strain_gauss_pairs:
            for batch_size in args.batch_sizes:
                for variant in args.variants:
                    cell_name = (
                        f"s{segments}-o{order}-g{gauss_points}-b{batch_size}"
                        f"-{variant}.json"
                    )
                    cell_output = args.cell_directory / cell_name
                    command = [
                        sys.executable,
                        str(benchmark),
                        "--device",
                        args.device,
                        "--topology",
                        args.topology,
                        "--segment-counts",
                        str(segments),
                        "--strain-gauss-pairs",
                        f"{order}:{gauss_points}",
                        "--batch-sizes",
                        str(batch_size),
                        "--operations",
                        "dynamics_terms",
                        "--variants",
                        variant,
                        "--repeats",
                        "1",
                        "--warmup-iterations",
                        "0",
                        "--output",
                        str(cell_output),
                    ]
                    environment = os.environ.copy()
                    if args.device == "cpu":
                        environment["OMP_NUM_THREADS"] = "1"
                        command = [
                            "taskset",
                            "-c",
                            str(args.cpu_core),
                            *command,
                            "--require-cpu-core",
                            str(args.cpu_core),
                        ]
                    print(
                        f"fresh cell: segments={segments} order={order} "
                        f"gauss={gauss_points} batch={batch_size} variant={variant}",
                        flush=True,
                    )
                    subprocess.run(command, check=True, env=environment)
                    cell = json.loads(cell_output.read_text(encoding="utf-8"))
                    if "metadata" not in aggregate:
                        aggregate["metadata"] = cell["metadata"]
                    aggregate["results"].extend(cell["results"])
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps(aggregate, indent=2), encoding="utf-8"
                    )

    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
