# Soromox Benchmarking Toolkit

Soromox ships development benchmarking CLIs under `tools/benchmarks`:

- `benchmark_system_methods.py` profiles individual model routines (forward kinematics,
  dynamics, etc.) to track JIT compile and steady-state execution costs.
- `benchmark_model_based_control.py` profiles the default model-based controller
  and controller-facing transformed-dynamics paths for PCS and GVS.
- `benchmark_derivative_paths.py` compares direct analytical derivative hooks,
  protected autograd fallbacks, and public APIs with custom JVPs enabled or disabled
  for PlanarPCS, PCS, and GVS systems.

The publication's batched simulation benchmark lives with the Section IVb paper
artifacts under `paper_results/secIVb_parallel_rollouts_gpu/`.

All benchmark generators share the same system registry and integration defaults, so adding a new
robot once makes it accessible throughout the benchmarking suite.

## Prerequisites

- Activate the `soromox` environment (or otherwise ensure the package is on `PYTHONPATH`).
- Install JAX (CPU or GPU build), Matplotlib, Seaborn (optional, for nicer plots),
  and any accelerator-specific drivers. GPU users should follow the
  [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Benchmarking individual system methods

The `benchmark_system_methods.py` CLI times a range of core routines for a sweep of
system sizes. Each measurement captures a cold call (compile + first execution) and
reports the median of synchronized warm calls as steady-state latency.

```bash
python tools/benchmarks/benchmark_system_methods.py \
  --device cpu \
  --systems articulated_soft_robot pendulum planar_pcs pcs \
  --segment-counts 1 3 5 7 \
  --coriolis-modes enabled disabled \
  --duration 2.0 \
  --solver-dt 5e-4 \
  --warmup-duration 1 \
  --execution-repeats 20 \
  --csv benchmarks/methods.csv \
  --plot benchmarks/methods.png
```

### Key options

- `--systems`: subset of available robots (defaults to all registered systems).
- `--segment-counts`: link/segment sweep; a fresh system instance is created per value.
- `--duration`, `--solver-dt` (`--dt` alias), `--save-dt`: integration controls when benchmarking
  `rollout_to`.
- `--device`: select `cpu`, `gpu`, or JAX's default automatic device choice.
  Explicit choices are applied before JAX is imported.
- `--warmup-duration`: synchronized, unmeasured execution time used to settle
  CPU frequency scaling and runtime worker threads before collecting samples.
- `--execution-repeats`: number of synchronized calls whose median is reported
  after the cold run and optional warmup.
- `--coriolis-modes`: benchmark enabled and/or disabled static model settings.
  Each result row records `consider_coriolis`, and plots use distinct lines for
  both modes.
- `--json` / `--csv`: export raw results for regression tracking.
- `--plot` / `--show-plot`: render Matplotlib summaries (compile vs. exec time).

### Interpreting the results

For each system/function pair the script prints:

1. Cold-call latency (compile + execution), synchronised via `block_until_ready()`.
2. Median warm-call latency, reflecting the steady-state cost once XLA caches
   the executable and rejecting transient scheduler outliers.
3. Derived compile time = cold − warm. Pure-Python methods show near-zero compile
   time but still track runtime cost.

The plots group results by system, highlighting how complexity evolves with segment
count for each tracked method.

### Pinning CPU benchmarks

On machines with heterogeneous CPU cores, process migration between performance
and efficiency cores can distort sub-millisecond measurements. Select the CPU
backend with `--device cpu` and, where the operating system supports it, pin the
whole benchmark process to a known core. On Linux, first inspect the topology and
maximum frequencies:

```bash
lscpu -e=CPU,CORE,MAXMHZ,ONLINE
```

Then run the benchmark with `taskset`; CPU 0 is only an example and should be
replaced with a performance core identified on the benchmark host:

```bash
taskset -c 0 python tools/benchmarks/benchmark_system_methods.py \
  --device cpu \
  --systems pcs \
  --segment-counts 1 4 8 16 \
  --warmup-duration 1 \
  --execution-repeats 20
```

Affinity applies to the Python process and its JAX worker threads. JSON and CSV
rows record the resolved `backend`, compact `cpu_affinity`, warmup duration, and
sample count so the execution context remains auditable. When comparing
revisions, use the same affinity and warmup settings for every run.

When comparing optional Coriolis paths, pass both modes. The exported rows make
enabled/disabled compile and warm-runtime ratios available for each size, and
the plots distinguish the two settings.

### Optional Coriolis reference measurement

The reference run used the CPU device, 1, 4, 8, and 16 links or segments, five
Gauss points for continuum systems, and 20 synchronized measured repetitions
after a one-second warmup per compiled case. On the benchmark host, CPUs 0-7
are the performance cores (5.5-5.7 GHz maximum), while CPUs 8-23 are the
efficiency cores (4.7 GHz maximum). The reference uses CPU 1 exclusively to
prevent migration both between core classes and among performance cores.
Determine the appropriate performance-core affinity from the topology of the
host being measured rather than copying this affinity to other machines. These
are the commands used to produce the reference:

```bash
JAX_ENABLE_X64=true taskset -c 1 \
python tools/benchmarks/benchmark_system_methods.py \
  --device cpu \
  --systems articulated_soft_robot \
  --segment-counts 1 4 8 16 \
  --methods dynamics_terms forward_dynamics rollout_to \
  --coriolis-modes enabled disabled \
  --execution-repeats 20 \
  --warmup-duration 1 \
  --duration 0.002 --solver-dt 0.0001 --save-dt 0.002 \
  --json /tmp/soromox-optional-coriolis-articulated-pcore.json \
  --csv /tmp/soromox-optional-coriolis-articulated-pcore.csv \
  --plot /tmp/soromox-optional-coriolis-articulated-pcore.png

JAX_ENABLE_X64=true taskset -c 1 \
python tools/benchmarks/benchmark_system_methods.py \
  --device cpu \
  --systems planar_pcs pcs gvs \
  --segment-counts 1 4 8 16 --gauss-points 5 \
  --methods dynamics_terms forward_dynamics rollout_to \
  --coriolis-modes enabled disabled \
  --execution-repeats 20 \
  --warmup-duration 1 \
  --duration 0.002 --solver-dt 0.0001 --save-dt 0.002 \
  --json /tmp/soromox-optional-coriolis-continuum-pcore.json \
  --csv /tmp/soromox-optional-coriolis-continuum-pcore.csv \
  --plot /tmp/soromox-optional-coriolis-continuum-pcore.png
```

Each size cell below is the enabled/disabled compile-time ratio followed by the
enabled/disabled median warm-runtime ratio. Values greater than one favor the
Coriolis-disabled path.

| System / method | 1 | 4 | 8 | 16 | Warm geometric mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| ArticulatedSoftRobot / `dynamics_terms` | 1.53x / 1.12x | 1.86x / 1.57x | 1.76x / 1.84x | 1.76x / 2.62x | 1.71x |
| ArticulatedSoftRobot / `forward_dynamics` | 1.48x / 1.87x | 1.30x / 1.16x | 1.30x / 1.40x | 1.29x / 1.27x | 1.40x |
| ArticulatedSoftRobot / `rollout_to` | 1.30x / 1.71x | 1.02x / 1.63x | 1.16x / 1.68x | 1.31x / 1.98x | 1.75x |
| PlanarPCS / `dynamics_terms` | 2.00x / 1.11x | 1.89x / 1.27x | 1.77x / 1.29x | 1.65x / 1.46x | 1.28x |
| PlanarPCS / `forward_dynamics` | 2.10x / 1.07x | 1.77x / 1.28x | 1.67x / 1.33x | 1.73x / 1.51x | 1.29x |
| PlanarPCS / `rollout_to` | 1.27x / 1.37x | 1.55x / 1.77x | 1.54x / 1.67x | 1.58x / 1.57x | 1.59x |
| PCS / `dynamics_terms` | 1.95x / 1.67x | 1.86x / 1.33x | 1.96x / 1.52x | 1.84x / 1.48x | 1.49x |
| PCS / `forward_dynamics` | 1.78x / 1.85x | 1.76x / 1.35x | 1.77x / 1.51x | 1.79x / 1.39x | 1.51x |
| PCS / `rollout_to` | 1.56x / 2.74x | 1.50x / 1.80x | 1.53x / 1.59x | 1.47x / 1.40x | 1.82x |
| GVS / `dynamics_terms` | 1.77x / 1.28x | 1.79x / 1.53x | 1.75x / 1.50x | 1.73x / 1.37x | 1.42x |
| GVS / `forward_dynamics` | 1.70x / 1.19x | 1.75x / 1.53x | 1.69x / 1.51x | 1.52x / 1.36x | 1.39x |
| GVS / `rollout_to` | 1.51x / 1.73x | 1.51x / 1.17x | 1.50x / 1.18x | 1.30x / 1.33x | 1.33x |

Across all requested methods and sizes, the per-system warm-runtime geometric
means are 1.61x for ArticulatedSoftRobot, 1.38x for PlanarPCS, 1.60x for PCS,
and 1.38x for GVS.

The per-size compile and warm timings are retained in the JSON and CSV artifacts.
The exported rows also record `backend`, `cpu_affinity`, the warmup duration,
and the sample count. Sub-millisecond calls are particularly sensitive to CPU
frequency scaling and scheduler placement, so performance comparisons should
retain both the warmup and explicit affinity.

For `articulated_soft_robot`, the method benchmark includes both the default
articulated-body forward dynamics path and a dense Jacobian-energy forward
dynamics solve (`forward_dynamics_dense`). Comparing these two cases is useful
for tracking ABA performance against the controller-facing dense dynamics API.

## Benchmarking model-based control

`benchmark_model_based_control.py` measures the checked-out implementation of
the computed-torque and configuration-/actuation-space feedforward terms, plus
the operational-space dynamics terms used by operational controllers. The
benchmark intentionally contains no legacy strategy switch: compare revisions
by running the same command in separate clean worktrees.

```bash
python tools/benchmarks/benchmark_model_based_control.py \
  --systems pcs gvs \
  --segment-counts 1 4 \
  --gauss-points 5 \
  --execution-repeats 50 \
  --json /tmp/model-based-control.json
```

Use `--methods` to select individual controller paths and `--csv` for a
tabular export. Each row reports derived compile time and synchronized warm
execution time for the default implementation on that revision.

## Benchmarking derivative paths

Use `benchmark_derivative_paths.py` when you want a direct runtime comparison
between direct analytical derivative implementations, protected autograd paths,
and public APIs with custom JVPs enabled or disabled. The benchmark covers
kinematic derivatives, Jacobian derivatives, and gradients of gravitational,
elastic, potential, kinetic, and total energy.

```bash
python tools/benchmarks/benchmark_derivative_paths.py \
  --systems planar_pcs pcs gvs \
  --segment-counts 1 2 4 8 16 32 \
  --execution-repeats 3 \
  --csv benchmarks/derivative-paths.csv \
  --json benchmarks/derivative-paths.json \
  --markdown-summary benchmarks/derivative-paths.md
```

Each row reports one `case`/`strategy` pair, including compile time, warm
execution time, ratios to the direct analytical and protected-autograd references,
and `max_abs_diff` / `max_rel_diff` sanity checks. The Markdown summary groups the
main speedup ratios by system and case.

## Benchmarking simulation batch scaling

`generate_benchmark_gpu.py` runs full simulations in parallel batches
using `jax.vmap`. It records both the per-environment speed
(`simulated_time / wall_time`) and the aggregate throughput
(`num_envs * simulated_time / wall_time`), so you can see whether each environment is
running faster than real-time *and* how many simulated seconds are produced per wall
second. Each configuration can export tables and plots spanning multiple segment counts.

```bash
python paper_results/secIVb_parallel_rollouts_gpu/code/generate_benchmark_gpu.py \
  --systems articulated_soft_robot pcs planar_pcs \
  --segment-counts 1 3 5 \
  --batch-sizes 1 2 4 8 16 32 64 \
  --duration 2.0 \
  --solver-dt 5e-4 \
  --csv /tmp/batch-scaling.csv \
  --plot /tmp/batch-scaling.png \
  --log-x --log-y
```

### Key options

- `--batch-sizes`: number of environments to launch per measurement.
- Shared `--systems`, `--segment-counts`, `--duration`, `--solver-dt` (`--dt` alias), `--save-dt`.
- `--noise-scale`: per-environment perturbation applied to `q`/`qd` to avoid feeding
  identical states to all replicas (helps stress vectorisation paths).
- `--repeats`, `--warmup-runs`: control timing stability.
- `--csv`, `--npz`, `--plot`, `--show-plot`, `--log-x`, `--log-y`: artifact and
  visualisation controls.

### Output and interpretation

For each combination of system, segment count, and number of environments the script reports:

- Wall-clock time averaged over the requested repeats.
- Mean simulated time per environment (i.e., the final timestamp returned by
  `rollout_to`).
- Per-environment speed ratio `simulated_time / wall_time`, total throughput
  `number_of_environments * simulated_time / wall_time`, and per-environment wall time. Ratios > 1
  indicate faster-than-real-time performance.

The generator's optional diagnostic plots show two stacked panels per system:
the top tracks per-environment
speed-up, while the bottom tracks aggregate throughput (total simulated time per
wall second). Each line corresponds to a segment count so scaling trends remain
easy to compare, with the horizontal axis representing the number of environments.

## Visualising existing runs

To revisit stored measurements (JSON or CSV) without re-running the benchmarks, use:

```bash
python tools/benchmarks/visualize_system_methods_results.py benchmarks/methods.json \
  --systems planar_pcs pcs \
  --functions rollout_to forward_dynamics \
  --output benchmarks/methods-focus.png
```

The helper mirrors the plotting style of `benchmark_system_methods.py` and accepts
optional filters for systems/functions. Pass `--show` to open a window interactively.

For the polished Section IVb publication plots, use:

```bash
python paper_results/secIVb_parallel_rollouts_gpu/code/plot_benchmark_gpu.py \
  --systems articulated_soft_robot planar_pcs pcs gvs \
  --segment-counts 1 2 4 8 16 32
```

## Extending the registry

Both CLIs share `tools/benchmarks/_benchmark_common.py`. To add a new system:

1. Implement a factory that builds the system for a requested segment/link count.
2. Provide a context builder returning representative `q`, `qd`, `u`, `tau_ext`, and
   any auxiliary data required by the benchmarking cases.
3. Register the new entry in the shared registry so it automatically becomes available
   to every benchmarking tool.

Following this pattern keeps benchmarks for future robot models (e.g., new
articulated variants or GVS variants) consistent and easy to maintain.
