# GVS Warp kernel investigation

## Scope and setup

This experiment is based on commit `60185cb6802a73117ed22c5eae8cdb0ee3d71b8e`
from `codex/lie-action-performance` and lives on `codex/gvs-warp-kernels`.
It uses Warp 1.16.0, JAX 0.11.0, FP64, and an RTX 5090. The legacy
Options 1--14 target four constant-strain, fixed-joint GVS segments, five Gauss
points and six recurrence cells per segment, and 24 active DoFs. Options
15--23 remove the compile-time segment/DoF dimensions and are tested with
higher-order strain bases and general joints. Autodifferentiation is
intentionally out of scope. `GVS` is currently a serial segment chain, so the
persistent kernels preserve that topology rather than introducing a rigid-body
tree algorithm.

All timings are synchronized wall times. The harness interleaves the tested
implementations and reports medians, which reduces clock, thermal, and ordering
bias. CUDA compilation and steady-state execution are reported separately.

## Implementations

| Option | Description |
|---|---|
| Baseline | `jax.jit(jax.vmap(robot.dynamics_terms))` |
| 1 | One parallel Warp launch per cell; JAX Lie terms and assembly |
| 2 | Six dependent cell launches captured in each segment graph |
| 3 | Warp contraction of JAX Lie arrays, then Option 2 |
| 4 | One persistent serial Warp thread per environment |
| 5 | One cooperative recurrence block plus one parallel assembly kernel per segment |
| 6 | General matrix-free Lie algebra, redundantly computed per output entry, then Option 2 |
| 7 | Option 6 Lie mapping plus Option 5 |
| 8 | General matrix-free Lie algebra with one owning thread per cell, then Option 2 |
| 9 | Option 8 Lie mapping plus Option 5 |
| 10 | Exact constant-strain Lie specialization, then Option 2 |
| 11 | Constant-strain specialization plus Option 5 |
| 12 | Option 8 plus exact fixed-joint elimination |
| 13 | Constant-strain and fixed-joint specializations, then Option 2 |
| 14 | Constant-strain and fixed-joint specializations plus Option 5 |
| 15 | Shape-generic two-point Magnus Lie terms and runtime-sized segment recurrence; general JAX joint propagation and JAX assembly |
| 16 | Option 15 with guarded order-zero and fixed-joint fast paths |
| 17 | Option 15 with persistent flattened model maps and shape-generic Warp joint Lie terms |
| 18 | Option 17 with one unique Warp owner per global `M`, `C qd`, and `G` output entry |
| 19 | Option 17 with one cooperative block per environment/segment, double-buffered recurrence, and online quadrature assembly |
| 20 | Option 19 with general joint-to-link state propagation fused into the segment block |
| 21 | One persistent runtime-looped block per environment for the complete serial chain |
| 22 | Option 21 plus a shape-generic Warp dense Cholesky solve for zero-input forward dynamics |
| 23 | Option 21 with model-map-guarded active-prefix recurrence and assembly |

## Phase-by-phase evaluation

### Phase 1: revalidate the existing segment graph

The previous conclusion survived the branch/worktree change. Option 2 remained
faster through batch 256 and reached parity at 1024.

| Environments | JAX | Option 2 |
|---:|---:|---:|
| 1 | 1.410 ms | 0.887 ms |
| 64 | 1.612 ms | 1.086 ms |
| 256 | 2.132 ms | 1.770 ms |
| 1024 | 5.169 ms | 5.269 ms |

Single-environment CPU: JAX 0.261 ms, Option 2 1.165 ms.

Decision: keep the segment graph as the reference Warp topology, but not as the
final implementation.

### Phase 2: cooperative segment recurrence and assembly

Option 5 keeps the current `6 x 24` Jacobian in a shared tile, advances all six
cells inside one cooperative block, writes the six recurrence states, and uses
a second parallel kernel to assemble the five quadrature contributions.

| Environments | JAX | Option 2 | Option 5 |
|---:|---:|---:|---:|
| 1 | 1.103 ms | 0.678 ms | **0.592 ms** |
| 64 | 1.273 ms | 0.884 ms | **0.795 ms** |
| 256 | 1.950 ms | **1.680 ms** | 1.687 ms |
| 1024 | **4.929 ms** | 5.167 ms | 5.296 ms |

At batch 64, Nsight Systems measured about 0.199 ms/evaluation for the four
recurrence plus four assembly kernels and 0.569 ms total GPU kernel time, down
from 0.630 ms for Option 2. The launch-saving topology helps small batches, but
block synchronization and the assembly layout lose throughput after saturation.

CPU: JAX 0.271 ms, Option 5 0.776 ms.

Decision: retain the cooperative topology for small and medium batches.

### Phase 3: matrix-free Lie algebra

The first mapping (Option 6) assigned one thread to each `6 x 24` output entry.
It recomputed the same strain, Magnus expansion, coefficients, and SE(3)
exponential 144 times per cell and became slower as the batch grew.

Option 8 instead assigns one thread to each cell. That thread computes shared
cell data once, applies the `ad` polynomial directly to the six active basis
columns, constructs the compact SE(3) exponential/adjoint, and emits the data
consumed by the recurrence.

| Environments | JAX | Option 2 | Option 8 |
|---:|---:|---:|---:|
| 1 | 1.141 ms | 0.756 ms | **0.683 ms** |
| 64 | 1.321 ms | 0.939 ms | **0.699 ms** |
| 256 | 1.969 ms | 1.819 ms | **1.075 ms** |
| 1024 | 5.260 ms | 5.407 ms | **2.553 ms** |

At batch 64, the winning Lie kernel costs about 0.071 ms/evaluation while the
24 segment-cell recurrence kernels cost about 0.221 ms. The bad Option 6 Lie
mapping alone cost about 0.525 ms/evaluation.

CPU: JAX 0.238 ms, Option 8 1.186 ms.

Decision: one cell owner is the correct general Lie mapping. Keep JAX on CPU.

### Phase 4: combine Phase 2 and Phase 3

Option 9 combines the one-cell-owner Lie kernel with cooperative segment
recurrence/assembly. It improves launch-bound cases but not saturated ones.

| Environments | Option 8 | Option 9 | Winner |
|---:|---:|---:|---|
| 1 | 0.685 ms | **0.613 ms** | Cooperative |
| 64 | 0.704 ms | **0.646 ms** | Cooperative |
| 256 | **1.133 ms** | 1.140 ms | Tie |
| 1024 | **2.360 ms** | 2.559 ms | Throughput |

Decision: use a batch-dependent mapping; fusion is not monotonically beneficial.

### Phase 5: fully fused Lie, recurrence, and quadrature

This phase failed the compile-feasibility gate. The strictly smaller monolithic
cooperative kernel containing only recurrence and quadrature did not finish Warp
CUDA compilation within a six-minute bound. Inlining the full Lie expansion
would make that same tile AST larger, so it was not meaningful to spend runtime
benchmark effort on an even less tractable version.

The practical split kernel from Phase 2 compiled and exposed the same on-chip
recurrence structure without the pathological monolithic code-generation path.

Decision: reject the monolithic Warp tile implementation with Warp 1.16.0.
Revisit only after decomposing it into smaller device functions/modules or after
Warp compiler improvements.

### Phase 6: exact model specializations

Two exact facts of this benchmark configuration are guarded at construction
time. They are not general properties of the `GVS` class: `_gvs_factory`
explicitly creates Legendre basis-order-zero links and `JointSpec(type="fixed")`
for every segment.

The benchmark-specific facts are:

- Order-zero constant strain gives `B_Z1 == B_Z2` and matching reference
  strains. Therefore the Magnus commutator and its derivative vanish,
  `Magnus = length * width * xi`, and `Magnus_basis = length * width * B`.
- All inter-segment joints are fixed: their adjoints are identity and all joint
  tangent, derivative, and velocity terms are zero. The link coordinates are
  contiguous six-DoF slices of `q` and `qd`.

For GVS basis order greater than zero, the strain varies spatially and the
general two-point Magnus terms must be retained. Likewise, articulated GVS
models with non-fixed joints must retain the general joint propagation. Those
models should use the general Option 8/9 path, not Options 13/14.

The Lie specialization lives in its own Warp module. This matters because Warp
compiles all kernels in a module together. A true empty-cache build showed that
the specialized Lie module itself took only 0.35--0.71 seconds; the original
static recurrence module was the real bottleneck at 102--106 seconds per FFI
variant. Cached module loading is about 1--4 ms.

The fixed-joint specialization was the larger win because it removed a
surprisingly expensive JAX joint path and its launches. Final focused results
are below.

### Phase 7: full-environment persistence

The existing one-thread-per-environment implementation was retested against the
Phase 6 winners.

| Environments | Persistent full environment | Best Phase 6 |
|---:|---:|---:|
| 1 | 19.093 ms | 0.533 ms |
| 64 | 28.234 ms | 0.639 ms |
| 256 | 33.156 ms | 0.885 ms |
| 1024 | 35.603 ms | 1.981 ms |

It serializes the complete recurrence and 24-DoF assembly while launching only
one thread per environment. The saved launches do not compensate for lost
parallelism and register/local-memory pressure.

Decision: reject full-environment persistence.

### Phase 8: shape-generic, forward-only production candidate

Option 15 retains the Phase 3 one-cell-owner idea, but changes the generated
program in four important ways:

- the two-point Magnus expansion remains active, so spatially varying GVS
  strains are supported;
- joints are evaluated through the general JAX joint path, so the Warp kernels
  do not assume fixed joints;
- each cell emits only its compact `6 x max_dof` tangent, and a runtime
  global-to-local lookup maps it into the system Jacobian;
- segment count, global DoF count, and local DoF count are runtime loop bounds
  rather than Python/Warp-unrolled constants.

Option 16 uses the same small module but enables the order-zero and fixed-joint
shortcuts only after construction-time guards prove they are valid. Both
options set `enable_backward=False`, which is appropriate for this explicitly
primal-only experiment. The old kernels received the same setting for a fair
compilation comparison.

This is the key compilation result. With a genuinely empty Warp and JAX cache,
Option 15 for four basis-order-one segments with revolute joints and batch 64
took:

| Stage | Time |
|---|---:|
| First Warp FFI variant, included in lowering | 0.761 s |
| JAX lowering total | 0.886 s |
| XLA executable compilation | 0.820 s |
| Second Warp FFI variant, first execution | 0.712 s |
| Lower + XLA compile | 1.706 s |
| Lower + XLA compile + first execution | **2.424 s** |

Once that shape-independent Warp module is cached, changing the link count or
basis order does not regenerate CUDA code. Every experiment below loaded the
same module hashes (`24c987e` and `5ac4abf`). Each JAX measurement used a fresh
persistent-cache directory.

| Segments | Basis order | Joint | DoFs | JAX lower + compile |
|---:|---:|---|---:|---:|
| 1 | 1 | revolute | 13 | 0.591 s |
| 2 | 1 | revolute | 26 | 0.715 s |
| 4 | 1 | revolute | 52 | 0.933 s |
| 8 | 1 | revolute | 104 | 1.210 s |
| 16 | 1 | revolute | 208 | 1.388 s |

The remaining growth is JAX/XLA compilation of progressively larger joint and
assembly graphs, not Warp CUDA source generation. At four revolute-jointed
segments, the basis-order sweep was similarly bounded:

| Basis order | DoFs | JAX lower + compile |
|---:|---:|---:|
| 0 | 28 | 0.795 s |
| 1 | 52 | 0.933 s |
| 2 | 76 | 1.077 s |
| 3 | 100 | 1.115 s |
| 5 | 148 | 1.158 s |

The guarded Option 16 path also compiled a 16-segment, 96-DoF fixed-joint model
in 1.103 seconds with the Warp module cached.

Disabling unused backward generation helped the old static module, but did not
make it suitable for runtime specialization. A fresh Option 13 build fell from
102--106 seconds to 14.84--15.00 seconds per FFI variant; reaching the first
completed call still took about 31.6 seconds. Options 13/14 should therefore be
ahead-of-time/cached peak-performance choices, not compiled per model shape.

### Phase 9: shape-generic Warp joint terms

Option 17 moves the general joint SE(3) exponential, left-Jacobian terms,
directional derivative, inverse adjoint, and inverse-adjoint derivative from
JAX into one runtime-sized Warp launch. Joint types are not compile-time
branches: the existing padded GVS joint basis encodes fixed, revolute,
prismatic, helical, cylindrical, planar, spherical, and free joints. Static
joint/link local-to-global maps and the varying-strain cell data are flattened
once into persistent arrays captured by the benchmark closure.

An empty-cache four-segment, basis-order-one, revolute-joint batch-64 build took
0.409 s to lower, 0.460 s to compile with XLA, and 0.344 s for the first
synchronized call, or **1.214 s** through the first completed result. A
16-segment mixed-joint model used the identical Warp module hashes and completed
in **1.662 s**. The extra size-dependent time remains in the JAX recurrence and
assembly graph rather than Warp CUDA generation.

Fifty cached, synchronized, interleaved GPU measurements for the four-segment,
basis-order-one revolute model were:

| Environments | JAX | Option 15 | Option 17 | Option 17 vs JAX |
|---:|---:|---:|---:|---:|
| 1 | 1.072 ms | 0.798 ms | 0.802 ms | 1.34x |
| 64 | 1.681 ms | 1.059 ms | **1.031 ms** | **1.63x** |
| 256 | 2.228 ms | 1.579 ms | **1.406 ms** | **1.58x** |
| 1024 | 6.041 ms | 4.727 ms | **4.121 ms** | **1.47x** |

The primary batch-64 improvement over Option 15 is modest (3%), but batch 256
improves 11% and the new joint/data path is required to fuse propagation in the
next phase. A mixed eight-segment model containing every non-fixed joint family
matched JAX within `9.17e-12`.

High-order basis evaluation remains unresolved. At basis order 5, Option 17
took 2.143 ms versus 2.000 ms for JAX at batch 64, and 4.087 ms versus 3.612 ms
at batch 256. Joint offload does not address the scalar link-coordinate loops
that dominate this case.

Decision: accept Option 17 as the new general-path foundation. Keep JAX on CPU,
and address high-order link evaluation in the later tiled/matrix-free phase.

### Phase 10: runtime-sized unique-owner assembly

Option 18 replaces only the JAX `M`, `C qd`, and `G` contractions with two
runtime-sized Warp kernels. Every output element has one owner, so no atomics
or nondeterministic reductions are used. Correctness remained at the Option 17
FP64 tolerance, but rereading the complete chain state for every output entry
was slower at every GPU batch.

| Environments | JAX | Option 17 | Option 18 |
|---:|---:|---:|---:|
| 1 | 1.233 ms | **0.978 ms** | 1.018 ms |
| 64 | 1.500 ms | **0.903 ms** | 0.953 ms |
| 256 | 2.240 ms | **1.405 ms** | 1.547 ms |
| 1024 | 6.126 ms | **4.205 ms** | 4.959 ms |

Single-environment CPU was 0.424 ms for JAX, 1.536 ms for Option 17, and
1.796 ms for Option 18.

Decision: reject the whole-chain owner-per-output assembly layout. Unique
ownership is useful, but the recurrence and quadrature must share live state.

### Phase 11: cooperative runtime-sized segment engine

Option 19 assigns one cooperative block to an environment/segment. It uses two
alternating global recurrence buffers, synchronizes cells inside the block,
assembles each quadrature contribution while its state is live, and returns
only the segment tip plus accumulated `M`, `C qd`, and `G`. Loop bounds remain
runtime values, so this does not regenerate source for a new model shape.

| Environments | JAX | Option 17 | Option 19 |
|---:|---:|---:|---:|
| 1 | 1.217 ms | **0.934 ms** | 1.248 ms |
| 64 | 1.526 ms | **0.861 ms** | 0.932 ms |
| 256 | 2.214 ms | 1.470 ms | **1.350 ms** |
| 1024 | 6.204 ms | 4.264 ms | **3.883 ms** |

It is a throughput mapping, not a small-batch mapping. At basis order 5 its
owner-per-mass-entry work becomes too large: 3.580 ms versus 1.895 ms for JAX
at batch 64, and 5.376 versus 3.589 ms at batch 256.

Decision: retain Option 19 for moderate DoF counts from roughly batch 256, but
continue using Option 17 for batch 1/64 and high-order links.

### Phase 12: joint propagation fused into each segment

Option 20 moves the general joint adjoint propagation into the Option 19 block.
It supports the same padded joint bases and does not branch on joint type in
generated source. The fusion boundary did not pay for its redundant block-local
joint work.

| Environments | JAX | Option 19 | Option 20 |
|---:|---:|---:|---:|
| 1 | 1.101 ms | **1.026 ms** | 1.084 ms |
| 64 | 1.518 ms | **0.954 ms** | 0.972 ms |
| 256 | 2.304 ms | 1.362 ms | **1.351 ms** |
| 1024 | 6.186 ms | **3.756 ms** | 3.924 ms |

Decision: reject Option 20 as a dispatch target. Keep the separate parallel
joint Lie kernel and fuse across segment boundaries instead.

### Phase 13: persistent complete-chain dynamics

Option 21 consumes the precomputed joint/cell Lie terms and traverses every
segment and cell in one runtime-looped cooperative kernel. It eliminates the
per-segment JAX additions and kernel boundaries while keeping general joints
and spatially varying strain.

| Environments | JAX | Option 17 | Option 19 | Option 21 |
|---:|---:|---:|---:|---:|
| 1 | 0.962 ms | **0.683 ms** | 0.906 ms | 0.925 ms |
| 64 | 1.649 ms | **1.071 ms** | 1.157 ms | 1.177 ms |
| 256 | 2.349 ms | 1.507 ms | 1.356 ms | **1.323 ms** |
| 1024 | 6.190 ms | 4.319 ms | 3.869 ms | **3.842 ms** |

The whole-chain launch is mildly better only after enough environments are
available. At batch 64, one block per environment cannot occupy all 170 SMs on
the RTX 5090.

Decision: retain Option 21 as infrastructure for the model-map sparsity phase,
not as the final unguarded implementation.

### Phase 14: forward solve

Option 22 adds a forward-only, runtime-sized dense Cholesky kernel and measures
complete zero-input forward dynamics,
`qdd = solve(M, -(C qd + G))`. The comparison uses JAX's compiled dense solve
after every other terms implementation.

| Environments | JAX end-to-end | Option 17 + JAX solve | Option 21 + JAX solve | Option 21 + Warp solve |
|---:|---:|---:|---:|---:|
| 1 | 1.381 ms | **1.055 ms** | 1.308 ms | 1.428 ms |
| 64 | 2.004 ms | **1.416 ms** | 1.509 ms | 1.525 ms |
| 256 | 2.645 ms | 1.710 ms | **1.598 ms** | 1.621 ms |
| 1024 | 6.930 ms | 4.905 ms | 4.360 ms | **4.186 ms** |

Forward acceleration differs from the JAX reference by about `1e-7`, because
the approximately `1e-12` term-ordering differences are amplified by the dense
mass solve. Both JAX and Warp solve the same well-formed SPD systems.

Decision: keep the JAX solve at the priority batch sizes. The Warp solve is
useful only at the non-priority batch-1024 end of this experiment.

### Phase 15: active-prefix persistent dynamics

For a serial GVS chain, segment `s` can depend only on coordinates introduced
at or before `s`. Option 23 derives the active prefix from the actual joint/link
gather maps. It enables the shortcut only when the active indices are exactly
`0..N_s-1`; otherwise that segment falls back to the full global DoF count.
The persistent buffers are initialized once, so newly introduced columns are
zero until their joint or link tangent becomes active.

This changes the dominant assembly work from `S * N^2` to
`sum_s(N_s^2)` without assuming fixed joints, order-zero strain, or a specific
joint family.

Four basis-order-one revolute-jointed segments (52 DoFs):

| Environments | JAX | Option 17 | Option 23 | Option 23 vs JAX |
|---:|---:|---:|---:|---:|
| 1 | 1.104 ms | **0.803 ms** | 0.875 ms | 1.26x |
| 64 | 1.481 ms | 0.853 ms | **0.699 ms** | **2.12x** |
| 256 | 2.266 ms | 1.481 ms | **0.968 ms** | **2.34x** |
| 1024 | 6.211 ms | 4.336 ms | **2.771 ms** | **2.24x** |

Single-environment CPU medians from 100 synchronized, interleaved repetitions:

| Basis order | DoFs | JAX CPU | Option 17 CPU | Option 23 CPU | Best Warp regression |
|---:|---:|---:|---:|---:|---:|
| 1 | 52 | **0.339 ms** | 1.131 ms | 0.424 ms | Option 23 is 25.1% slower |
| 5 | 148 | **0.703 ms** | 1.636 ms | 1.523 ms | Option 23 is 116.6% slower (2.17x runtime) |

Four basis-order-five revolute-jointed segments (148 DoFs):

| Environments | JAX | Option 17 | Option 23 | Best Warp regression |
|---:|---:|---:|---:|---:|
| 64 | **2.011 ms** | 2.162 ms | 2.326 ms | Option 17 is 7.5% slower |
| 256 | 3.612 ms | 4.189 ms | **3.021 ms** | Option 23, 1.20x |

Eight basis-order-one segments cycling through revolute, prismatic, helical,
cylindrical, planar, spherical, and free joints:

| Environments | JAX | Option 17 | Option 23 | Option 23 vs JAX |
|---:|---:|---:|---:|---:|
| 1 | **1.788 ms** | 1.847 ms | 2.350 ms | 0.76x |
| 64 | 3.224 ms | 2.980 ms | **2.471 ms** | 1.30x |
| 256 | 5.329 ms | 5.984 ms | **3.111 ms** | 1.71x |
| 1024 | 17.893 ms | 18.206 ms | **10.744 ms** | 1.67x |

All joint-family and term correctness checks remain below `9.17e-12`.

True empty-cache compilation is both comfortably below ten seconds and flat
with segment count. Times include compilation of both Warp FFI variants across
lowering and first execution.

| Model | Option 23 lower | XLA compile | First call | Through first result | JAX lower + compile |
|---|---:|---:|---:|---:|---:|
| 4 segments, order 1, revolute | 0.456 s | 0.037 s | 0.408 s | **0.901 s** | 1.635 s |
| 16 segments, order 1, mixed | 0.476 s | 0.045 s | 0.421 s | **0.942 s** | 1.626 s |

Both shapes compile identical Warp module hashes (`15724be`/`5072bdb`,
`24c987e`/`5ac4abf`, and `a9298da`/`a6f1840`). The larger model changes runtime
array sizes but not CUDA source. This is the desired compilation scaling law.

Decision: accept Option 23 as the primary serial-chain GPU candidate. Dispatch
Option 17 at very small batches, use JAX for high-order batch 64, and retain JAX
on CPU. The next optimization target for large mixed-joint models is avoiding
the full global joint-tangent materialization and fusing cell Lie preprocessing
with active-prefix recurrence, while keeping these runtime loops.

## Fixed-model steady-state results

These are medians of 50 cached, synchronized, interleaved measurements from the
latest lookup implementation. Option 14 is the cooperative mapping; Option 13
is the throughput mapping.

| Environments | JAX | Option 13 | Option 14 | Option 15 | Option 16 | Best |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.928 ms | 0.353 ms | **0.290 ms** | 0.499 ms | 0.381 ms | Option 14, 3.20x |
| 64 | 1.228 ms | 0.556 ms | **0.448 ms** | 0.663 ms | 0.546 ms | Option 14, 2.74x |
| 256 | 1.969 ms | 0.805 ms | **0.740 ms** | 0.952 ms | 0.782 ms | Option 14, 2.66x |
| 1024 | 4.990 ms | **1.771 ms** | 1.942 ms | 2.449 ms | 1.849 ms | Option 13, 2.82x |

Option 16 is the practical runtime-compiled substitute for the old specialized
options. It is 2% faster than Option 13 at batch 64 and 4% slower at batch 1024,
while avoiding shape-specific Warp compilation. Option 14 retains a meaningful
small-batch advantage and is worth distributing as a prebuilt/cached kernel for
the exact supported shape.

The dispatch crossover should be measured on each target GPU; for this RTX 5090
and fixed shape, use Option 14 through 256 environments and Option 13 at 1024.

Single-environment CPU, 100 measurements:

| Implementation | Runtime | Relative to JAX |
|---|---:|---:|
| JAX | **0.241 ms** | 1.00x |
| Option 14 | 0.520 ms | 0.46x |
| Option 13 | 1.051 ms | 0.23x |

Keep the compiled JAX path for CPU.

## General GVS results

Option 15 was evaluated on four segments with basis order 1 and a revolute
joint at every segment. This case has 52 DoFs and violates both assumptions
behind Options 13/14.

| Environments | JAX | Option 15 | Speedup |
|---:|---:|---:|---:|
| 1 | 1.352 ms | **1.034 ms** | 1.31x |
| 64 | 1.755 ms | **1.227 ms** | 1.43x |
| 256 | 2.082 ms | **1.452 ms** | 1.43x |
| 1024 | 6.204 ms | **4.769 ms** | 1.30x |

Single-environment CPU is still a JAX case: 0.377 ms for JAX versus 1.289 ms
for Option 15. At basis order 5 and batch 64, Option 15 remained correct but
was slower than JAX (2.936 versus 2.091 ms); runtime-sized scalar loops lose to
XLA's dense contractions as local strain dimension grows. The scalable path
solves compilation scaling, but does not make Warp universally faster.

## Correctness

Every implemented option is compared with the JAX baseline before timing. For
the final specialized options, the largest observed absolute errors were:

- inertia: `2.13e-13`;
- `C(q, qd) qd`: `5.06e-14`;
- gravity: `3.72e-12`.

For the general basis-order-one/revolute model, the largest observed error was
`3.44e-12`; at basis order 5 it was `2.97e-12`. A 16-segment Option 16 check was
within `2.09e-11`.

The harness rejects errors above `2e-9`. The larger difference compared with
the earlier recurrence-only kernels comes from changed FP64 evaluation order in
the direct matrix-free Lie polynomials, not from a changed model.

## Final profiler evidence

Nsight Systems 2025.5.2 captured 20 warm CUDA-graph evaluations with node
tracing enabled.

At batch 64, Option 14 uses about 0.285 ms GPU kernel time/evaluation:

- cooperative recurrence: 0.096 ms;
- cooperative assembly: 0.104 ms;
- constant-strain Lie terms: 0.051 ms;
- remaining JAX/XLA kernels: about 0.034 ms.

At batch 1024, Option 13 uses about 1.51 ms GPU kernel time/evaluation. Its 24
cell-recurrence kernels consume about 1.01 ms (67%), constant-strain Lie terms
consume 0.113 ms, and the remaining contractions/assembly consume the rest.
This explains both the remaining fusion opportunity and why the serial
full-environment response is the wrong way to pursue it.

Nsight Compute requires privileged performance counters on this host. An
unprivileged capture returns `ERR_NVGPUCTRPERM`; the Codex sandbox cannot invoke
`sudo`, so a user-run `sudo ncu` command is included below.

## Reproduction

From this worktree:

```bash
cd /home/mstolzle/src/soromox/.worktrees/gvs-warp-kernels
env WARP_CACHE_PATH=/tmp/soromox-warp-cache \
  PYTHONPATH="$PWD/src:$PWD" \
  /home/mstolzle/src/soromox/.venv/bin/python \
  tools/benchmarks/benchmark_gvs_warp.py \
  --device gpu --batch-sizes 1 64 256 1024 --repeats 50 \
  --options option_13 option_14 option_15 option_16 \
  --output /tmp/soromox-gvs-warp-final-gpu.json
```

Use `JAX_PLATFORMS=cpu --device cpu --batch-sizes 1` for CPU timing.

General higher-order/revolute runtime and compilation:

```bash
env WARP_CACHE_PATH=/tmp/soromox-warp-cache \
  PYTHONPATH="$PWD/src:$PWD" \
  /home/mstolzle/src/soromox/.venv/bin/python \
  tools/benchmarks/benchmark_gvs_warp.py \
  --device gpu --segment-count 4 --basis-order 1 --joint-type revolute \
  --batch-sizes 1 64 256 1024 --repeats 50 \
  --options option_17 option_23 \
  --output /tmp/soromox-gvs-warp-option23-gpu.json

env WARP_CACHE_PATH=/tmp/soromox-warp-cache-cold \
  JAX_COMPILATION_CACHE_DIR=/tmp/soromox-jax-cache-cold \
  PYTHONPATH="$PWD/src:$PWD" \
  /home/mstolzle/src/soromox/.venv/bin/python \
  tools/benchmarks/benchmark_gvs_warp.py \
  --device gpu --segment-count 4 --basis-order 1 --joint-type revolute \
  --batch-sizes 64 --compile-option option_23 --compile-repeats 1 \
  --output /tmp/soromox-gvs-warp-option23-compile.json
```

Measure uncached JAX/XLA GPU compilation with:

```bash
env JAX_ENABLE_COMPILATION_CACHE=false \
  WARP_CACHE_PATH=/tmp/soromox-warp-cache \
  PYTHONPATH="$PWD/src:$PWD" \
  /home/mstolzle/src/soromox/.venv/bin/python \
  tools/benchmarks/benchmark_gvs_warp.py \
  --device gpu --batch-sizes 1 64 256 1024 \
  --compile-option baseline --compile-repeats 3 \
  --output /tmp/soromox-gvs-jax-cold-compile-gpu.json
```

Nsight Systems example:

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=/tmp/soromox-gvs-warp-option14-b64 \
  /usr/bin/env WARP_CACHE_PATH=/tmp/soromox-warp-cache \
  PYTHONPATH="$PWD/src:$PWD" \
  /home/mstolzle/src/soromox/.venv/bin/python \
  tools/benchmarks/benchmark_gvs_warp.py \
  --device gpu --batch-sizes 64 \
  --profile-option option_14 --profile-iterations 20
```

Nsight Compute example for the batch-1024 winning Lie kernel:

```bash
sudo env WARP_CACHE_PATH=/tmp/soromox-warp-cache \
  PYTHONPATH="$PWD/src:$PWD" \
  /opt/nvidia/nsight-compute/2025.4.1/ncu \
  --force-overwrite --profile-from-start off --graph-profiling node \
  --section LaunchStats --section Occupancy --section SpeedOfLight \
  --section MemoryWorkloadAnalysis --section SourceCounters \
  --launch-count 1 \
  --kernel-name 'regex:^constant_strain_cell_terms_kernel.*$' \
  --export /tmp/soromox-gvs-warp-option13-b1024-lie \
  /home/mstolzle/src/soromox/.venv/bin/python \
  tools/benchmarks/benchmark_gvs_warp.py \
  --device gpu --batch-sizes 1024 \
  --profile-option option_13 --profile-iterations 1
```

## Recommendation

Use the following forward-only dispatch on the measured RTX 5090:

- CPU: compiled JAX.
- General serial-chain GVS, very small GPU batches: Option 17.
- General serial-chain GVS, order 1, batch 64 and above: Option 23.
- High strain-basis order at batch 64: JAX; remeasure the Option 17/23 crossover
  from batch 256 on each target GPU.
- Zero-input/applied-force forward solve at batch 64/256: keep the JAX dense
  solve after the selected Warp terms path. Use the Warp Cholesky experiment
  only after a target-specific large-batch crossover measurement.
- Exact order-zero/fixed-joint models reused many times: the prebuilt Option 14
  small/medium-batch kernel remains fastest for that narrower model class.

Option 23 is the main result: it supports spatially varying GVS strain and all
current joint families, exceeds 2x JAX for the four-segment order-one priority
batches, and reaches the first result in under one second on both the measured
four- and 16-segment shapes. It does not generate source proportional to the
number of segments or DoFs.

The remaining general-path opportunities are narrower and evidence-based:
retain joint tangents in local coordinates instead of materializing a
`6 x global_DoFs` tensor, combine active-prefix propagation with cell Lie
preprocessing, and introduce a multi-block assembly mapping for high-order
batch 64. These should preserve the runtime-looped modules; the rejected static
monolithic tile must not return to the first-use compilation path.
