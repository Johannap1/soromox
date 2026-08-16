# Section Vd: Control-Gain Optimization

This case compares gain optimization for collocated actuation-space control and
synergistic operational-space control. Each generator writes a MAT result under
its controller-specific directory in `data/`; the plotter combines both files.

Run the two optimizations without GUI rendering:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/control_gain_optimization_with_collocated.py \
  --result-dir /tmp/soromox-secVd/collocated --num-iters 100 --no-show --no-render
uv run python paper_results/secVd_control_gain_optimization/code/control_gain_optimization_with_synergistic.py \
  --result-dir /tmp/soromox-secVd/synergistic --num-iters 100 --no-show --no-render
```

To replace the canonical case-local data, omit `--result-dir` and pass
`--force`. Diagnostic plots are always written to `--output-dir`; Open3D
visualization is skipped with `--no-render`.

Pressing Ctrl-C during an optimization preserves the finite iterations that
finished before the interrupted evaluation. The generator then continues
through its normal plot and data-saving path.

Recreate the canonical comparison from the committed MAT files:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/plot_control_gain_optimization.py
```

Render standalone tracking diagnostics plus MP4 and GIF animations from the
saved MAT trajectories:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/render_control_gain_optimization_animations.py \
  --force
```
