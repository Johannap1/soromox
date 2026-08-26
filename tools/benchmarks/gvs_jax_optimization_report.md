# Compile-bounded JAX GVS dynamics evaluation

## Conclusion

Several ideas behind PR #173 transfer usefully to JAX, but specializing one
JAX recurrence body per segment does not. The exact active-prefix implementation
delivers the largest warmed-runtime gains, but its StableHLO graph and compiler
time grow linearly with segment count. The selected implementation recovers most
of the useful reduction savings with four reusable prefix-width branches inside
one fixed outer `lax.scan`.

The final design:

- reuses the spatial velocity already carried by the recurrence;
- uses the compact one-row strain-basis representation;
- partitions the ordered segments into at most four equally populated prefix
  buckets;
- puts `lax.switch` only around the three global `B`, `C @ qd`, and `G`
  reductions; and
- retains the original fixed-width path below 96 total DOFs, where GPU branch
  overhead is not consistently amortized.

After all four branches exist, the StableHLO operation count is independent of
segment count. The final K=4 implementation therefore restores the baseline
compilation scaling law while retaining substantial CPU and batched-GPU runtime
gains.

The branch `codex/pr173-jax-active-prefix-snapshot` preserves the exact unrolled
implementation at commit `38ad507b90`. All subsequent experiments remain in the
history of `codex/pr173-jax-evaluation`, including the rejected alternatives.

## Measurement method

- JAX 0.11.0 with 64-bit arithmetic.
- CPU: Intel Core Ultra 9 285K. Every CPU timing used exactly high-performance
  logical core 1 through `taskset -c 1`, with `OMP_NUM_THREADS=1`. The benchmark
  validates the affinity and aborts unless it is exactly `(1,)`.
- GPU: NVIDIA GeForce RTX 5090, driver 595.84.
- Homogeneous systems used 1, 4, 8, 16, and 32 segments. Strain-order/Gauss-point
  pairs were 0/5, 1/5, 3/7, and 5/9; Gauss points therefore increase with strain
  order.
- CPU used one environment. GPU batches were 1, 64, 256, and 1024, with the
  32-segment endpoint measured through batch 256.
- Runtime measurements are synchronized medians after 8--10 warm-up calls,
  normally over 15 GPU or 30 CPU samples.
- Every compilation-scaling cell ran in a fresh Python/JAX process. The reported
  compile time is `lowered.compile()`; lowering, first execution, StableHLO text
  size, operation count, `case` count, and `while` count were recorded separately.
- Candidate values and JVP directional derivatives were compared with the
  original fixed-shape implementation before timing.

The main runner is `benchmark_gvs_jax_dynamics.py`. The fresh-process
orchestrator is `run_gvs_jax_compile_scaling.py`.

## Increasing-complexity experiments

| Phase | Change evaluated | Main observation | Decision |
|---:|---|---|---|
| 1 | Reuse recurrence velocity; compact strain-basis rows; keep full-width reductions | Small CPU gains and modest GPU gains, but the expensive global reductions still operate on all future DOFs | Keep both inexpensive building blocks |
| 2 | K=2/4/8 fixed prefix-width branches only around global reductions | Large-batch runtime improves while the number of compiled branches is bounded | Keep |
| 3 | Uniform-width, equal-segment-count, and padding-cost-optimal bucket placement | Cost-optimal reduces nominal padding but is 1--4% slower on GPU, apparently because its irregular GEMM widths are less efficient; equal-count is simple and robust | Select equal-count |
| 4 | Separate local-cell branches by link DOF or by `(DOF, Gauss points)` | StableHLO grows from about 3,151 to 4,726--4,891 operations and runtime is neutral to 5% slower | Reject |
| 5 | Also bucket the 6-by-N recurrence transforms | Adds 91--171 StableHLO operations and regresses runtime by about 3--7% on CPU and 1--13% on GPU | Reject |
| 6 | Dynamic tiled reductions with widths 32/64/128 | Keeps a fixed graph, but many small GEMMs regress CPU/GPU runtime by 19--45% | Reject |
| 7 | Increase the branch budget to K=16 and select a dispatch threshold | K=16 buys only another 2--3% over K=8 while doubling branch count; small systems do not amortize the switch | Select K=4 and a 96-DOF guard |

### What happened to grouping by equal local DOFs?

Segments with equal local DOF/Gauss shapes were evaluated both in alternating
and contiguous layouts. A separate compiled branch per local shape is not useful:
it duplicates most of the cell evaluator and substantially increases the graph.
It also does not solve prefix padding, because a segment's global active prefix
depends on every preceding segment, not only its own local DOF count.

Contiguous local-shape grouping can still help indirectly: equal-count prefix
buckets then tend to end at a shape boundary. The final heterogeneous results
below show slightly better runtime for the grouped layout, without adding a new
JAX branch class.

## Selecting the branch budget

The representative order-5/GQ-9 results below show the Pareto tradeoff. Times are
fresh-process compile seconds / warmed term-assembly milliseconds.

| Device/workload | Fixed scan | K=2 | K=4 | K=8 | Exact prefix |
|---|---:|---:|---:|---:|---:|
| CPU, S=16, B=1 | 1.97 / 13.04 | 2.07 / 9.49 | 2.18 / 8.34 | 2.48 / 8.11 | 7.90 / 5.88 |
| GPU, S=16, B=256 | 3.35 / 146.98 | 3.58 / 112.00 | 4.54 / 96.39 | 7.17 / 89.28 | 14.99 / 57.97 |
| GPU, S=16, B=1024 | 4.39 / 576.79 | 5.21 / 429.40 | 6.64 / 369.35 | 10.77 / 346.14 | 19.83 / 219.88 |
| GPU, S=32, B=256 | 4.40 / 1004.56 | 5.13 / 752.84 | 6.83 / 646.87 | 10.65 / 606.36 | not run |

K=4 captures roughly 59--68% of the exact-prefix runtime gain in representative
large cases, while its graph has only one `case` with four branches. K=8 provides
only another 4--7% runtime reduction but roughly doubles the incremental compile
cost. K=4 was therefore selected.

## Compilation scaling

### StableHLO graph law

The graph-size result is the central design criterion.

| Segments | CPU fixed scan | CPU K=4 | CPU exact prefix | GPU B=256 fixed | GPU B=256 K=4 | GPU B=256 exact |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2,933 | 3,147 | 3,260 | 3,121 | 3,347 | 3,480 |
| 8 | 2,933 | 3,147 | 4,048 | 3,121 | 3,347 | 4,364 |
| 16 | 2,933 | 3,147 | 5,624 | 3,121 | 3,347 | 6,132 |
| 32 | 2,933 | 3,147 | 8,776 | 3,121 | 3,347 | not run |

The exact path adds 197 CPU or 221 batched-GPU StableHLO operations per segment.
K=4 adds a fixed 214 CPU or 226 batched-GPU operations over the baseline after
four segments and then plateaus. Both fixed and K=4 graphs retain two `while`
operations; K=4 has one bounded `case` operation.

StableHLO text still grows with tensor shapes and embedded constants. That is
not loop-body duplication: operation count stays fixed, and the compiler sees
the same recurrence/control-flow topology at every larger segment count.

### Fresh-process CPU compile time

Cells are fixed / K=4 / exact compile seconds.

| Segments | order 1 / GQ 5 | order 5 / GQ 9 |
|---:|---:|---:|
| 4 | 1.878 / 2.043 / 2.697 | 1.922 / 2.110 / 2.846 |
| 8 | 1.867 / 2.052 / 4.350 | 1.969 / 2.162 / 4.488 |
| 16 | 1.967 / 2.118 / 7.858 | 1.988 / 2.260 / 7.898 |
| 32 | 1.937 / 2.106 / 15.255 | 2.294 / 2.468 / 15.326 |

Linear fits over 4--32 segments quantify the change:

| CPU path | order-1 compiler slope | order-5 compiler slope | order-5 lowering + compiler slope |
|---|---:|---:|---:|
| Fixed scan | 0.0026 s/segment | 0.0132 s/segment | 0.0209 s/segment |
| K=4 | 0.0023 s/segment | 0.0128 s/segment | 0.0223 s/segment |
| Exact prefix | 0.4502 s/segment | 0.4472 s/segment | 0.4856 s/segment |

Thus the K=4 CPU scaling law is indistinguishable from the fixed scan at this
resolution. At 32 segments, K=4 compiles in 2.11--2.47 seconds versus
15.26--15.33 seconds for exact prefix. The order-5 K=4 overhead over fixed is
only 8%, versus 568% for exact prefix.

### Fresh-process GPU segment and batch scaling

For order 1, fixed / K=4 compiler seconds are:

| Segments | batch 1 | batch 256 | batch 1024 |
|---:|---:|---:|---:|
| 4 | 1.19 / 1.50 | 1.34 / 1.60 | 1.46 / 1.90 |
| 8 | 1.26 / 1.54 | 1.48 / 1.95 | 1.45 / 1.99 |
| 16 | 1.23 / 1.73 | 1.46 / 2.03 | 1.78 / 2.22 |
| 32 | 1.27 / 1.70 | 1.48 / 1.95 | not run |

The K=4 order-1 fits over 4--32 segments are only 0.007--0.008 seconds per
segment for batches 1/256, within fresh-process noise and close to the fixed
scan's 0.002--0.003 seconds per segment. At batch 1024, the 4--16 slope is
0.027 seconds per segment, compared with the previous exact-prefix slope of
about 0.39 seconds per segment.

The high-order case exposes shape-driven GPU code generation. Cells below are
fixed / K=4 / exact compiler seconds.

| Segments | batch 1 | batch 256 | batch 1024 |
|---:|---:|---:|---:|
| 4 | 1.74 / 3.55 / 3.80 | 2.64 / 4.06 / 4.50 | 2.81 / 4.89 / 5.82 |
| 8 | 1.72 / 3.25 / 6.45 | 2.88 / 4.30 / 8.26 | 3.09 / 4.70 / 8.84 |
| 16 | 1.68 / 3.30 / 12.30 | 3.15 / 4.54 / 14.99 | 4.42 / 6.64 / 19.83 |
| 32 | 1.88 / 3.27 / not run | 4.36 / 6.83 / not run | not run |

At batch 1, K=4 is flat from 4 through 32 segments while exact prefix adds
about 0.71 seconds per segment. At batch 256, the 4--32 slopes are 0.061 seconds
per segment for fixed and 0.100 for K=4; exact prefix adds about 0.88 seconds per
segment over 4--16. At batch 1024, the 4--16 slopes are 0.139 seconds per segment
for fixed, 0.160 for K=4, and 1.17 for exact prefix.

The remaining K=4 growth at large batch/order occurs despite a constant 3,347
StableHLO operations. It is therefore shape/layout/GEMM code-generation cost,
which also exists in the fixed scan, rather than a segment-unrolled graph. At
16 segments, K=4 cuts exact-prefix GPU compile time by 67--73%, depending on
batch size.

## Final warmed runtime

Negative regressions are reported rather than hidden. The public dispatch keeps
the original path for fewer than 96 total DOFs; this makes 4-segment order 0/1
and 8-segment order 0 unchanged.

### Single-environment pinned CPU term assembly

Cells show speed-up over the fixed scan.

| Segments | order 0 / GQ 5 | order 1 / GQ 5 | order 3 / GQ 7 | order 5 / GQ 9 |
|---:|---:|---:|---:|---:|
| 4 | unchanged | unchanged | 15.4% | 28.5% |
| 8 | unchanged | 16.2% | 26.9% | 30.4% |
| 16 | 24.9% | 29.5% | 36.0% | 36.4% |
| 32 | 28.6% | 28.9% | 33.2% | 32.6% |

Representative order-5 absolute medians are 12.65 -> 8.04 ms at 16 segments
and 100.95 -> 68.04 ms at 32 segments. At 32 segments the exact unrolled path
is slower than K=4 for orders 3 and 5, presumably because the much larger
compiled program is less effectively optimized.

### GPU term assembly

Cells show speed-up for batches 1 / 64 / 256 / 1024. The 32-segment rows stop at
batch 256.

| Segments | order / GQ | Speed-up by batch |
|---:|:---|---:|
| 4 | 0 / 5 | unchanged |
| 4 | 1 / 5 | unchanged |
| 4 | 3 / 7 | 3.5% / 9.2% / 13.2% / 17.7% |
| 4 | 5 / 9 | -15.3% / 8.3% / 19.1% / 27.6% |
| 8 | 0 / 5 | unchanged |
| 8 | 1 / 5 | 1.3% / 10.2% / 12.4% / 13.7% |
| 8 | 3 / 7 | 0.8% / 12.1% / 21.2% / 27.9% |
| 8 | 5 / 9 | -13.4% / 20.5% / 31.9% / 31.8% |
| 16 | 0 / 5 | -0.9% / -2.8% / 7.8% / 13.5% |
| 16 | 1 / 5 | 3.2% / 5.8% / 15.8% / 22.9% |
| 16 | 3 / 7 | 0.4% / 22.3% / 30.5% / 30.4% |
| 16 | 5 / 9 | 3.7% / 31.5% / 35.7% / 36.4% |
| 32 | 0 / 5 | -6.6% / 10.9% / 16.6% |
| 32 | 1 / 5 | 4.7% / 17.0% / 23.1% |
| 32 | 3 / 7 | 11.6% / 28.4% / 30.7% |
| 32 | 5 / 9 | 13.9% / 34.0% / 34.1% |

The switch overhead is visible for GPU batch 1, especially at 4/8 segments and
high order. The target multi-environment workloads consistently improve once
there is enough work: at 16 segments/order 5, medians fall from 146.76 to
94.32 ms at batch 256 and from 578.00 to 367.33 ms at batch 1024. At 32
segments/order 5/batch 256 they fall from 1.007 s to 0.664 s.

### Complete forward dynamics

The dense inertia solve reduces the fraction of time spent in term assembly but
the selected gains remain visible.

| Segments | order / GQ | CPU B=1 | GPU B=64 | GPU B=256 |
|---:|:---|---:|---:|---:|
| 4 | 1 / 5 | unchanged | unchanged | unchanged |
| 4 | 5 / 9 | 32.2% | 10.4% | 17.2% |
| 8 | 1 / 5 | 16.6% | 2.2% | 10.2% |
| 8 | 5 / 9 | 28.5% | 10.8% | 24.1% |
| 16 | 1 / 5 | 27.3% | 7.3% | 13.6% |
| 16 | 5 / 9 | 31.2% | 20.4% | 26.8% |

The pinned-CPU 16-segment/order-5 median is 14.46 -> 9.95 ms. The GPU
batch-256 median is 193.17 -> 141.38 ms.

### Heterogeneous segment shapes

The heterogeneous case alternates order-3/GQ-7 and order-5/GQ-9 links, or puts
each equal-shape class contiguously. Results are term-assembly medians.

| Layout | CPU B=1 | GPU B=256 |
|---|---:|---:|
| Alternating | 8.79 -> 5.63 ms (35.9%) | 103.59 -> 68.97 ms (33.4%) |
| Equal shapes grouped | 8.85 -> 5.28 ms (40.3%) | 103.54 -> 65.79 ms (36.5%) |

Grouping equal local shapes is therefore modestly useful as a model/layout
property, but compiling a separate evaluator branch per local shape is not.

## Correctness and implementation notes

- The exact-prefix implementation remains available as the `exact_prefix`
  benchmark variant and on the snapshot branch for direct comparisons.
- K=4 and exact-prefix outputs match the fixed scan for `B`, `C @ qd`, and `G`
  to the existing float64 tolerances.
- JVP directional derivatives match for velocity reuse, compact basis, all
  K=2/4/8 bucket policies, and exact prefix.
- A heterogeneous alternating-shape system is covered explicitly.
- Public-path tests cover the fixed small-system dispatch and the bounded K=4
  dispatch.
- The benchmark supports homogeneous, alternating, and grouped topologies and
  refuses unpinned CPU measurements when `--require-cpu-core` is supplied.
- The dedicated GVS system suite passes all 88 tests. A second targeted run of
  every remaining test file that explicitly references GVS passes all 112 tests.
  The repository-wide suite was stopped at 63% at the user's request; its five
  earlier failures did not reproduce in either GVS-focused run.

## Recommendation

Use the selected K=4 equal-count prefix buckets as the default for systems with
at least 96 total DOFs. This preserves a fixed recurrence graph and removes the
fundamental per-segment compilation-law regression. Retain the fixed scan for
small models, and keep exact prefix as an opt-in benchmark/research variant for
long-running, throughput-dominated jobs where cold compilation is irrelevant.
