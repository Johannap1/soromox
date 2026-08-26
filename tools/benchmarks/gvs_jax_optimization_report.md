# JAX GVS dynamics optimization evaluation

This report evaluates which structural ideas related to PR #173 transfer usefully
to the JAX GVS dynamics path. The work was performed on branch
`codex/pr173-jax-evaluation`, based on `origin/main` at
`014bc376a7555234af407d0fb88951952fa380e7`.

## Method

- JAX 0.11.0 with 64-bit arithmetic.
- CPU: Intel Core Ultra 9 285K. Every reported CPU measurement used one
  high-performance core with an exact affinity mask (`taskset -c 1`) and
  `OMP_NUM_THREADS=1`. The benchmark aborts if the affinity is not exactly core 1.
- GPU: NVIDIA GeForce RTX 5090 (32,607 MiB), driver 595.84.
- The main sweep used 1, 4, 8, and 16 segments; strain-order/Gauss-point pairs
  0/5, 1/5, 3/7, and 5/9; CPU batch 1; and GPU batches 1, 64, 256, and 1024.
  The largest 16-segment GPU cases used batches through 256.
- Inputs are deterministic, nonzero configurations and velocities. Timings are
  synchronized device execution medians after compilation and warm-up.
- All candidate variants were checked against the original fixed-shape path for
  values and JVP directional derivatives before timing.

The reproducible runner is `benchmark_gvs_jax_dynamics.py`. It records lowering,
compilation, first-execution, median, minimum, p90, and all raw execution samples
as JSON. In addition to dynamics assembly, it can benchmark complete forward
dynamics, including elasticity, damping, actuation, and the inertia solve.

## One-by-one results

Negative percentages below mean faster execution.

| Idea | CPU result | GPU result | Decision |
|---|---:|---:|---|
| Reuse the velocity already propagated by the recurrence instead of recomputing `J @ qd` | median -1.1%, range -3.6% to -0.2% | median about -4%; reverse-order rerun also about -4% | Keep inside the multi-segment recurrence |
| Cache invariant transforms and preweight quadrature masses | centered near 0%, mixed signs | order-sensitive, with regressions up to 8% in the reverse run | Reject; XLA already folds/fuses this effectively |
| Specialize every segment recurrence and reduction to its active DOF prefix | up to -49.8%; 16-segment median -40.4% | up to -60.2%; 16-segment median -45.5% | Keep for multi-segment models |
| Replace dense 6x6 adjoint products with matrix-free cross-product actions | median -0.4%, range -3.3% to +1.2% | median +0.7%, range -3.4% to +5.2% | Reject; no robust improvement |
| Store each structurally one-row strain-basis column as `(row, value)` and gather adjoint columns directly | median +0.2%, range -1.2% to +1.4% | median -3.4% and -4.1% in forward/reverse runs; only one small regression in each run | Keep as a modest GPU improvement; CPU-neutral |
| Assemble only the upper inertia triangle and mirror it | +95.5% to +2563.2% | neutral only in the smallest case; otherwise up to +51.4% | Reject; gathers/scatters cost more than the saved products |
| Fuse per-segment Lie-term evaluation into the causal recurrence | +6.3% to +45.7% | +6.5% to +27.0% | Reject; loses useful parallelism and increases unrolling |

The public path deliberately retains the original fixed-shape implementation for
a one-segment model: there is no shrinking prefix before the only segment, so the
extra specialization cannot provide its core benefit. Multi-segment models use
the retained active-prefix and compact-basis changes.

## Final steady-state dynamics results

Each cell reports speed-up over the original fixed-shape JAX implementation.
The CPU cells also show baseline and optimized median milliseconds.

| Segments | order 0 / GQ 5 | order 1 / GQ 5 | order 3 / GQ 7 | order 5 / GQ 9 |
|---:|---:|---:|---:|---:|
| 1 | unchanged | unchanged | unchanged | unchanged |
| 4 | 8.9% (0.110 → 0.100) | 7.8% (0.129 → 0.119) | 17.9% (0.226 → 0.186) | 29.2% (0.412 → 0.291) |
| 8 | -4.3% (0.206 → 0.215) | 7.3% (0.308 → 0.285) | 32.5% (0.831 → 0.561) | 42.0% (1.804 → 1.047) |
| 16 | 9.8% (0.602 → 0.543) | 34.0% (1.292 → 0.852) | 46.9% (4.747 → 2.521) | 51.1% (12.520 → 6.122) |

The one observed CPU regression is the low-order 8-segment case (-4.3%). It is
small in absolute terms (9 microseconds), while gains grow rapidly with strain
order and segment count.

| Segments | order / GQ | GPU batch 1 | batch 64 | batch 256 | batch 1024 |
|---:|:---|---:|---:|---:|---:|
| 4 | 0 / 5 | 11.1% | 0.7% | 10.7% | 6.8% |
| 4 | 1 / 5 | 4.4% | 4.5% | 18.1% | 8.8% |
| 4 | 3 / 7 | 18.2% | 12.9% | 25.7% | 28.3% |
| 4 | 5 / 9 | 3.0% | 22.1% | 26.6% | 36.2% |
| 8 | 0 / 5 | 2.7% | 11.8% | 16.3% | 11.1% |
| 8 | 1 / 5 | 10.3% | 14.0% | 21.8% | 26.5% |
| 8 | 3 / 7 | 7.2% | 28.3% | 39.9% | 47.2% |
| 8 | 5 / 9 | 5.2% | 33.8% | 48.7% | 51.1% |
| 16 | 0 / 5 | -0.8% | 18.8% | 22.4% | — |
| 16 | 1 / 5 | 5.5% | 19.9% | 40.9% | — |
| 16 | 3 / 7 | 2.3% | 45.3% | 57.3% | — |
| 16 | 5 / 9 | 4.7% | 55.0% | 60.0% | — |

The fixed-joint cross-check showed the same behavior: 2–44% CPU gains and
2–49% GPU gains over the representative 4/8-segment, order-1/5 cases.

## End-to-end forward dynamics

The assembly gains remain visible after adding elastic and damping forces and
solving the dense inertia system.

- Pinned CPU: 4–34% faster for 4–16 segments in the representative order-1/5
  sweep. The 16-segment/order-5 median fell from 14.248 ms to 9.817 ms (31.1%).
- GPU batch 64/256: 8–38% faster. The 8-segment/order-5/batch-256 median fell
  from 33.448 ms to 20.649 ms (38.3%).

## Compilation tradeoff

Active-prefix specialization unrolls the segment recurrence at trace time. In
fresh processes for an 8-segment, order-5, 9-Gauss-point model, the fixed-shape
reference below includes the already accepted recurrence-velocity reuse:

| Device/workload | Fixed-shape compile | Active-prefix compile | Compact compile | Fixed-shape run | Final run |
|---|---:|---:|---:|---:|---:|
| CPU, batch 1 | 1.933 s | 4.351 s | 4.443 s | 1.778 ms | 1.058 ms |
| GPU, batch 256 | 2.622 s | 7.913 s | 8.142 s | 25.637 ms | 13.434 ms |

For this shape, the extra compilation cost breaks even after roughly 3,500 CPU
calls or 450 batched GPU calls. The optimization is therefore aimed at compiled
simulation/training workloads with repeated evaluations. Short-lived workloads
that compile a multi-segment shape and evaluate it only a handful of times can
have worse wall-clock latency despite faster steady-state execution.

## Correctness coverage

- Exact/practical-tolerance comparisons of `B`, `C @ qd`, and `G` against the
  existing public matrix/force implementations.
- JVP comparisons of all retained candidate paths against the original path.
- Mixed Monomial, Legendre, and Fourier bases; mixed revolute, planar, and
  helical joints; fixed and revolute performance models.
- Parameter updates that change geometry/length and therefore refresh cached
  compact basis values.
- Rotational strain-basis length scaling, including transformed inertia,
  convective, and gravity terms.
