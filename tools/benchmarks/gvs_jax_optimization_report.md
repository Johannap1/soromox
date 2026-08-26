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

Active-prefix specialization unrolls the segment recurrence at trace time. A
second compilation-only sweep measured every shape in a fresh Python/JAX
process, avoiding in-process compilation caches. `compile` below is the time
spent in `lowered.compile()`; JAX lowering is reported separately afterward.
The 1-segment public paths are identical, so their differences establish a
rough 2–10% fresh-process noise floor.

### CPU segment scaling

CPU compilation was pinned to the same high-performance core as execution.
Cells show fixed-shape / optimized compile seconds.

| Segments | order 1 / GQ 5 | ratio | order 5 / GQ 9 | ratio |
|---:|---:|---:|---:|---:|
| 1 | 1.384 / 1.386 | 1.00x | 1.450 / 1.420 | 0.98x |
| 2 | 1.805 / 1.907 | 1.06x | 1.905 / 1.975 | 1.04x |
| 4 | 1.875 / 2.679 | 1.43x | 1.936 / 2.825 | 1.46x |
| 8 | 1.869 / 4.408 | 2.36x | 1.952 / 4.461 | 2.28x |
| 16 | 1.912 / 7.661 | 4.01x | 1.968 / 7.887 | 4.01x |

For 2–16 segments, the fixed-shape compiler time is effectively constant:
approximately `1.82 + 0.006 S` seconds at order 1 and
`1.91 + 0.004 S` seconds at order 5. The optimized path is linear in segment
count with an almost perfect fit: `1.07 + 0.413 S` seconds at order 1 and
`1.12 + 0.422 S` seconds at order 5 (`R² = 1.000` for both). Strain order has
little effect on CPU compilation.

Lowering the fixed-shape CPU graph takes about 0.17 seconds for 2–16 segments.
Optimized lowering grows to 0.35 seconds at 16 segments/order 1 and 0.50 seconds
at 16 segments/order 5. Including lowering, the 16-segment cold compilation
ratio is 3.85–3.92x rather than 4.01x.

### GPU segment and batch scaling

For order 1 / GQ 5, cells show fixed-shape / optimized compile seconds and the
ratio in parentheses.

| Segments | batch 1 | batch 64 | batch 256 | batch 1024 |
|---:|---:|---:|---:|---:|
| 1 | 0.84 / 0.92 (1.10x) | 0.97 / 0.99 (1.02x) | 1.11 / 1.14 (1.02x) | 1.32 / 1.39 (1.05x) |
| 2 | 0.99 / 1.11 (1.12x) | 1.14 / 1.21 (1.06x) | 1.16 / 1.19 (1.03x) | 1.41 / 1.49 (1.06x) |
| 4 | 1.23 / 1.85 (1.50x) | 1.34 / 1.97 (1.47x) | 1.35 / 2.07 (1.54x) | 1.47 / 2.44 (1.66x) |
| 8 | 1.25 / 3.35 (2.69x) | 1.23 / 3.55 (2.88x) | 1.64 / 3.48 (2.13x) | 1.46 / 4.07 (2.78x) |
| 16 | 1.29 / 6.26 (4.84x) | 1.48 / 6.35 (4.28x) | 1.40 / 6.41 (4.57x) | 1.80 / 6.98 (3.88x) |

At fixed batch size, optimized order-1 compilation adds 0.37–0.39 seconds per
segment (`R² >= 0.998`), whereas the fixed-shape scan adds only 0.01–0.03
seconds per segment. A two-factor fit over 2–16 segments gives:

`T_optimized ≈ 0.339 + 0.364 S + 0.037 log2(B) + 0.0015 S log2(B)` seconds

with `R² = 0.994`. Segment count is therefore the dominant compile axis. Batch
size changes tensor shapes but does not unroll the graph; from batch 1 to 1024,
compile time usually grows by only 0.2–0.7 seconds at order 1.

Higher strain order introduces a stronger batch/code-generation interaction:

| Segments | batch 1 | batch 256 | batch 1024 |
|---:|---:|---:|---:|
| 4 | 1.77 / 3.86 (2.18x) | 2.69 / 4.53 (1.68x) | 3.09 / 5.72 (1.85x) |
| 8 | 1.80 / 6.56 (3.63x) | 2.98 / 8.17 (2.74x) | 3.20 / 8.87 (2.77x) |
| 16 | 1.63 / 12.25 (7.50x) | 3.37 / 15.50 (4.61x) | 4.60 / 21.60 (4.69x) |

These order-5 optimized slopes are about 0.70 seconds/segment at batch 1,
0.91 at batch 256, and 1.36 at batch 1024. Repeats of the 8- and 16-segment,
batch-1024 points agreed within 6%, confirming the large-shape increase is not
an ordering artifact.

### Why the scaling changes

StableHLO inspection separates graph growth from shape-driven code generation:

- At batch 1, the fixed-shape graph stays at 2,933 StableHLO operations from
  2 through 16 segments. The optimized graph has 3,260 operations at 4
  segments, 4,048 at 8, and 5,624 at 16—an exactly linear increment of about
  197 operations per added segment after segment 4.
- At batch 1024, the corresponding counts are 3,120 operations for every
  fixed-shape model and 3,480 / 4,364 / 6,132 for optimized 4 / 8 / 16-segment
  models, or about 221 operations per added segment.
- Increasing from order 1 to order 5 does not change those operation counts,
  but it enlarges constants, tensor types, GEMMs, and layout/fusion search. For
  example, the 16-segment/batch-1024 fixed-shape StableHLO text grows from
  0.62 MB to 1.61 MB. This explains why high order and large batches increase
  GPU compilation even though they do not add recurrence bodies.

The implementation trades a reusable fixed-width scan body for one statically
specialized body per segment. That makes compiler work approximately linear in
segment count, while execution avoids arithmetic on all not-yet-active DOFs.

### Compile amortization

Combining isolated compile overhead with warmed execution medians gives the
number of calls needed to recover the additional compile time:

- CPU order 5: about 7,400 calls at 4 segments, 3,300 at 8, and 925 at 16.
  Low-order CPU cases can require 13,000–114,000 calls because each call saves
  only microseconds.
- GPU order 5, batch 256: about 1,215 calls at 4 segments, 414 at 8, and 131 at
  16. At batch 1024 this falls to about 342 calls at 4 segments and 113 at 8.
- GPU batch 1 generally requires 10,000–44,000 calls. Although steady-state
  execution is faster, compilation dominates short single-batch jobs.

The optimization is consequently best for long-running simulations, training,
or batched rollouts. If cold-start latency matters, the current default is a
clear tradeoff: multi-segment steady-state throughput improves, but compilation
scales linearly rather than remaining nearly constant.

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
