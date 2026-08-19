# Section Vd: Control-Gain Optimization

This case compares six-start gain optimization for collocated
actuation-space control and synergistic operational-space control.

> [!WARNING]
> The committed archives and comparison figures are single-start results from
> the current workflow. They do not reproduce the published six-start Section
> Vd study and must not be treated as recovered paper results.

## Canonical result format

Each optimizer writes only `optimization_results.npz`, using schema version 2.
The archive carries a real multi-start batch axis of width `B`, and that axis
appears **only** on quantities that genuinely vary per start:

| field | shape |
| --- | --- |
| `history_loss`, `history_time`, `history_finite_mask` | `(iterations, B)`, `(iterations,)`, `(iterations, B)` |
| `history_Kp`, `history_Ki`, `history_Kd` | `(iterations, B, m)` |
| `init_Kp`, `init_Ki`, `init_Kd` | `(B, m)` |
| `q_ts_init/best`, `qd_ts_init/best` | `(B, timesteps, dofs)` |
| `u_ts_init/best` | `(B, timesteps, actuators)` |
| `x_ts_init/best` | `(B, timesteps, 6)` |
| `best_iteration` | `(B,)` |
| `best_batch`, `batch_size` | scalars |
| `t_ts`, `x_des_ts`, `q_des_ts` | `(timesteps,)`, `(timesteps, 6)`, `(timesteps, dofs)` |

The time grid and the reference are shared by every start and so carry no batch
axis at all. Pose vectors use

```text
[rotation-vector x, y, z, Cartesian position x, y, z].
```

`q_ts_best` is the authoritative trajectory for reconstructing the complete
robot geometry. `x_ts_best` provides a directly validated full end-effector
pose trajectory. Schema-v1 archives, legacy MAT archives, `animation_data.pkl`,
and `intermediate_metrics.json` are not supported.

`completed_iterations` records the actual run length and is validated against
the stored histories. `is_placeholder` identifies development-only archives;
omit `--placeholder` when producing full results.
**Per-iteration trajectory histories are deliberately absent.** Storing every
start's rollout at every iteration would reach several gigabytes at 100
iterations and six starts, so the archive keeps each start's initial and best
rollouts only. The loss and gain histories remain complete.

`init_Kp` / `init_Ki` / `init_Kd`, together with `init_seed`, `init_scheme`, and
`init_spread`, record the initialization directly rather than leaving it to be
inferred from the history. The legacy results were unrecoverable precisely
because this was never written; see the provenance section below.

A start whose evaluation becomes non-finite is frozen rather than aborting the
whole run, and `history_finite_mask` marks exactly which entries are real. Every
start must still reach at least one finite iterate: one that was non-finite from
its first evaluation has no trajectory worth archiving and indicates an
initialization that should be resampled with a different `--init-seed`.

Selection is recorded and re-checked on load. `best_iteration` is each start's
own lowest-loss iteration and `best_batch` is `argmin_b min_i loss[i, b]`,
derived from **that archive's** losses alone -- the defect behind issue #128 was
a single index taken from one method and reused for the other.

The placeholder metadata is stored as `is_placeholder=true`. Omit
`--placeholder` when producing full results.

## Optimization

The two optimization programs only run optimization and generate their NPZ.
Both optimize six independent gain initializations at once, matching the width
of the original Section Vd results:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/control_gain_optimization_with_collocated.py \
  --num-iters 100 --force
uv run python paper_results/secVd_control_gain_optimization/code/control_gain_optimization_with_synergistic.py \
  --num-iters 100 --force
```

`--num-iters` defaults to 100. Pass another positive value when intentionally
running a shorter or longer optimization.

Use `--result-dir` to stage results elsewhere. A run is saved only if every
requested iteration completed and every start produced at least one finite
iterate, so an interrupted run cannot replace an archive.

### Initialization

Start 0 is always the nominal gain set, so a single-start run stays a strict
subset of a batched one. Starts 1 to `B-1` scale the nominal by one factor per
gain, drawn log-uniformly from `[1 / spread, spread]` -- gains are scale
parameters, so the neighbourhood is multiplicative rather than additive, and
each start remains describable by three numbers. Every run prints its table and
records `init_Kp`, `init_Ki`, `init_Kd`, `init_seed`, `init_scheme`, and
`init_spread` in the archive.

| flag | default | meaning |
| --- | --- | --- |
| `--batch-size` | `6` | independent starts `B` |
| `--init-seed` | `0` | PRNG seed behind the sampled starts |
| `--init-scheme` | `log_uniform_v1` | sampling scheme, recorded in the archive |
| `--init-spread` | `3.0` | multiplicative half-width for `log_uniform_v1` |

The alternative `legacy_uniform` scheme draws absolute values from the dormant
box found in the pre-packaging generator (`Kp` in `[10, 60]`, `Ki` in `[5, 40]`,
`Kd` in `[2, 5]`, seed 35). It exists only to test a recovered original
generator against, and is not the recovered initialization; see the provenance
section below.

### Device

Device selection defaults to `--device auto`, which uses the CPU for a single
start and prefers the GPU for `B > 1`. Note that JAX names the CUDA backend
`cuda`, not `gpu`; `JAX_PLATFORMS=gpu` is rejected outright. An automatic GPU
preference therefore requests `cuda,cpu` and warns if it lands on the CPU, while
an explicit `--device gpu` requests `cuda` alone and fails loudly rather than
running somewhere the user did not ask for. Device selection must happen before
JAX is imported, since JAX ignores `JAX_PLATFORMS` afterwards, which is why the
CLI and initialization modules import JAX lazily.

### Collocated saturation

The collocated controller integrates tendon-length errors in metres and uses
the unit-preserving saturation

```text
sat(e) = tanh(gamma * e) / gamma,
gamma = 1 / e_sat.
```

The default is `e_sat = 10 mm` (`gamma = 100 1/m`) and can be changed with
`--integral-error-saturation-scale` in metres.

## Plotting

The standalone plotter is the only comparison-figure entrypoint. It requires
schema version 1, plots the single run without synthetic min/max bands, and
writes both the canonical PDF and PNG:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/plot_control_gain_optimization.py \
  --force
```

## Robot rendering

The renderer reconstructs the optimized robot from `q_ts_best`, validates it
against the stored full pose, resamples the dense rollout to the requested FPS,
and follows the paper rendering style in Viser. The solid coral body is the
current robot. Collocated control additionally shows the desired configuration
as a pale blue-gray wireframe, without task-space markers. Synergistic control
uses a translucent coral body so that the current task position in magenta and
the larger desired task position in green remain visible. A dotted trail shows
the desired path only when the reference is time-varying; the animation does
not reveal the robot's future actual trajectory. It writes MP4 files and, when
requested, GIF previews:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/render_control_gain_optimization_animations.py \
  --method both --fps 24 --gif --force
```

Use `--method collocated` or `--method synergistic` to render one controller.
The CLI also exposes the Viser port/browser and recording timeout settings.

## Version-control policy

Canonical comparison figures (`plot_control_gain_opt.pdf` and `.png`) are kept
in normal Git. Canonical optimization archives and robot-rendering MP4/GIF
files are also versioned, but routed through Git LFS because full optimization
and rendering runs can make them substantially larger. Directories matching
`outputs/*_diagnostics/` are local scratch artifacts from the removed embedded
diagnostic workflow and are intentionally ignored.

The current five-iteration placeholder set contains the two NPZ archives, the
regenerated comparison PDF/PNG, and genuine Viser MP4/GIF robot captures from
those archives. Chart-animation files from the former renderer are not
canonical outputs.

## Provenance of the legacy six-batch results

> [!NOTE]
> **Provisional.** This section records what could be established from the
> artifacts reachable today. A machine that may hold the original generator is
> expected to become available in the week of 2026-08-24; recheck the open
> points below against it before treating this as final. Tracked as issue #154.

The Section Vd results published in the preprint were produced by a **six-start
batched** optimization that no committed script can reproduce. The generators in
this directory are single-start, and until the change described below they wrote
a batch axis pinned to one. This section answers the questions raised in issue
#154 and separates what is recovered from what is lost.

### Where the evidence comes from

Two artifacts, both preserved in Git history on `origin/main` at
`data/{collocated,synergistic}/optimization_results.mat`:

| variable | collocated | synergistic |
| --- | --- | --- |
| `history_loss` | `(100, 6)` | `(100, 6)` |
| `t_ts` | `(6, 50001)` | `(6, 50000)` |
| `q_ts_init`, `q_ts_best` | `(6, 50001, 6)` | — |
| `x_ts_init`, `x_ts_best` | — | `(6, 50000, 3)` |
| `q_des_ts` | `(50001, 6)` | — |
| `x_des_ts` | — | `(1, 3)` |

Axes are `(iterations, batch)` for the loss and `(batch, timesteps, components)`
for the trajectories. The `50001` / `50000` step counts match a `save_dt =
solver_dt = 1e-4` grid over `[0, 5] s`.

Both files carry the MAT header `Platform: PCWIN64`, which MATLAB writes and
SciPy does not (`scipy.io.savemat` writes `nt` or `posix`), and both share a
single creation timestamp. **The archives were serialized by MATLAB**, so even
the Python program that produced the underlying run wrote some intermediate
format that was not kept.

### Recovered

**Execution methodology — vectorized, not independent runs.** The batch is a
`vmap` over both the optimization variables and the optimizer state, giving `B`
independent optimizers stepping in lockstep. Gradients are never averaged across
the batch. This is *multi-start*, not *multiple shooting*: the rollout is a
single forward integration with one initial condition, no segment boundaries and
no continuity constraints.

**Selection.** The best batch is the one attaining the lowest loss over all
iterations,

```text
best_batch = argmin_b  min_i  history_loss[i, b]
```

which evaluates to `3` for **both** methods in the committed data. That
coincidence is what allowed the plotting bug in issue #128 to go unnoticed.

**Aggregation, and the published numbers.** `docs/research.md` reports 62% and
57% loss reductions "relative to their initial median values". The median is
taken across the batch at iteration zero:

```text
improvement = 1 - min(history_loss[-1]) / median(history_loss[0])
```

This reproduces the published figures exactly, which pins the metric:

The legacy archives were replaced by the NPZ workflow and are no longer checked
out, so retrieve them from history first:

```bash
case=paper_results/secVd_control_gain_optimization
for m in collocated synergistic; do
  git show "origin/main:$case/data/$m/optimization_results.mat" > "/tmp/secVd-$m.mat"
done
```

```python
import numpy as np, scipy.io as sio

for method in ("collocated", "synergistic"):
    loss = sio.loadmat(f"/tmp/secVd-{method}.mat")["history_loss"]
    improvement = 1.0 - np.min(loss[-1]) / np.median(loss[0])
    print(f"{method:12s} {100 * improvement:5.2f} %")
# collocated   62.62 %
# synergistic  57.82 %
```

`q_ts_init` / `x_ts_init` hold every batch's iteration-zero rollout and
`q_ts_best` / `x_ts_best` every batch's own best rollout. That is what the
median line and the min/max bands in the comparison figure are computed from.

### Not recovered

**The six initial gain sets, and the seed that produced them.** The archives
contain no gain history whatsoever — only losses, the time grid, and the
initial/best trajectories. Nothing in them constrains `Kp`, `Ki` or `Kd`.

The nearest surviving script,
`OPTso_closedloop_pcs3at_ws_trajectory_multishoot.py` in the author's
pre-packaging tree, is **not** the generator. It has `batch_size = 1`,
`num_iters = 3` and a `0.5 s` horizon rather than the archives' `5 s`, and its
`jax.random.PRNGKey(35)` is split but never consumed, so its multi-start branch
never executes. It does contain the dormant sampling box

```text
Kp in [10, 60],  Ki in [5, 40],  Kd in [2, 5]     # i.i.d. uniform, PRNGKey(35)
```

and its setpoint `x_des = [-0.055, -0.094, 0.118]` matches the synergistic
archive's `x_des_ts` verbatim. So the archives came from an **uncommitted
variant** of that script family. The box above is preserved as the
`legacy_uniform` initialization scheme for comparison, but it belongs to a
different controller parametrization and must not be presented as the recovered
value.

**A structural reason the initialization could never have been read back.** In
that script the loss at history index `j` is evaluated *before* the optimizer
update whose parameters are appended at index `j`. Losses and gains are
therefore off by one, and index `0` already holds one Yogi step away from the
nominal — the true initialization is never written at all. This is the
misalignment reported in issue #129, and it is why the schema described above
now stores `init_Kp` / `init_Ki` / `init_Kd` and the sampling metadata
explicitly, rather than expecting them to be inferable from the history.

### Consequences for regenerated results

Because the initialization is lost, regenerated Section Vd results **will not
reproduce 62% / 57%**. Two further changes since the original run also move the
numbers: both methods now share the `u_constant = [0.5, 0.2, 0.1]` setpoint, and
the collocated integral-error saturation moved from `gamma = 10` to the
unit-preserving `gamma = 100 1/m`. The figures in `docs/research.md` are left
untouched until the recovery check above resolves.
