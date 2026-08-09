# Port the identified AM-I Support GVS model from MATLAB/SoRoSim to soromox

Rewrites `examples/simulation/pcs/simulate_isupport_exp.py` to reproduce the
identified AM-I Support model from the MATLAB GVS pipeline
(`am_isupportGVS`, run `T1_fc28`, config hash `0d2ecd93`), and verifies the two
engines against each other.

**The two simulations agree to 0.028 mm RMS on a 625 mm peak-to-peak tip swing
(0.0044 %), 36x inside the 1 mm acceptance threshold, and soromox reproduces the
MATLAB tip RMSE against VICON — 40.128 mm — to three decimals.**

---

## Result: square wave (bag `2026-06-08-12-11-25`, 27.60–42.60 s)

```
PARITY (TIP) : soromox vs MATLAB
RMS difference        :   0.0276 mm     (x 0.0115 | y 0.0060 | z 0.0244)
max difference        :   0.0896 mm     (sample 519)
mean signed offset    :  +0.0010  +0.0029  +0.0097 mm
reference peak-to-peak:    625.4 mm
RMS / peak-to-peak    :   0.0044 %
VERDICT               : PASS  (threshold 1.0 mm)

PARITY (MID) : RMS 0.0081 mm | max 0.0260 mm | 0.0041 % of 196.3 mm -> PASS
```

Every metric scored against VICON matches the MATLAB reference:

| metric | MATLAB | soromox |
|---|---|---|
| tip RMSE (3D) | 40.128 mm | 40.128 mm |
| tip RMSE per axis (x, y, z) | 20.089 / 14.486 / 31.573 mm | 20.089 / 14.486 / 31.572 mm |
| static offset at t0 | 2.918 mm | 2.918 mm |
| mean amplitude error | 18.270 mm (6.9 %) | 18.269 mm (6.9 %) |
| mean phase lag | −3.60° | −3.60° |
| spectrum 1 / 3 / 5 Hz | −0.7 % / +14.3 % / −81.6 % | −0.7 % / +14.3 % / −81.6 % |

These reproduce the `T1_fc28` row of the MATLAB `REPORT.md` line for line.

### MATLAB vs soromox vs VICON

The two simulation curves are visually indistinguishable; both track the
measurement well except at the resonance (see "Known limitation" below).

![Tip traces](https://raw.githubusercontent.com/tud-phi/soromox/am_isupport/examples/simulation/pcs/amisupport_dataset/results/square_tip_traces.png)

### Parity residual (soromox − MATLAB)

Sub-0.1 mm throughout, against a 1 mm threshold. The residual grows mildly
across the window and carries no structure tied to the 1 Hz drive — the
signature of independent solver step-size histories, not a model difference.

![Parity residual](https://raw.githubusercontent.com/tud-phi/soromox/am_isupport/examples/simulation/pcs/amisupport_dataset/results/square_parity_residual.png)

### Per-axis error against VICON

Both engines make the *same* errors, of the same size, at the same times.

![Per-axis error](https://raw.githubusercontent.com/tud-phi/soromox/am_isupport/examples/simulation/pcs/amisupport_dataset/results/square_axis_error.png)

### Trajectory overlay

![3D overlay](https://raw.githubusercontent.com/tud-phi/soromox/am_isupport/examples/simulation/pcs/amisupport_dataset/results/square_tip_3d.png)

---

## What changed

### No model literals

Every parameter — geometry, material constants, per-actuator gains, chamber
clock, channel permutation, pressure lag, fit window, solver tolerances — is
read from a `.mat` exported by `python_exporter.m` on the MATLAB side.
Re-identify, re-export, re-run, and the two stay in step.

### Three geometry values were wrong

| | was | is | why |
|---|---|---|---|
| base segment length | 41 mm | **42 mm** | `g_ini` carries a 36 mm x-offset on top of the 6 mm short interface |
| soft segment radius | 30 mm | **28.6 mm** | `updateAmISupport.m` sets `VLinks(2).r = 28.6e-3` |
| soft segment density | 6471 kg/m³ | **2721.2 kg/m³** | the identified value |

Folding the 36 mm `g_ini` offset into segment 0 is exact, not an approximation:
`g_ini`'s rotation is a roll about x, so it leaves the x axis invariant and the
offset is collinear with the first link. The extra mass has no dynamic effect —
segment 0 is proximal to every strain DOF, so nothing can move it.

Rest-pose FK now lands on `s_tip = 435.0000 mm`, `s_mid = 249.0000 mm`,
identical to MATLAB to 0.000000e+00 mm.

### Layout

MATLAB's `r s r r s r` chain collapses to soromox's alternating layout by
merging the two consecutive rigid links (6 mm short + 21 mm long interface) into
one 27 mm body:

| seg | kind | L [mm] | r [mm] | ρ [kg/m³] |
|---|---|---|---|---|
| 0 | rigid | 42.000 | 30.000 | 1210.0 |
| 1 | soft | 180.000 | 28.600 | 2721.2 |
| 2 | rigid | 27.000 | 30.000 | 1210.0 |
| 3 | soft | 180.000 | 28.600 | 2721.2 |
| 4 | rigid | 6.000 | 30.000 | 1210.0 |

### Actuation now carries the identified gains

```
soromox pressure fed to the solver = g_i · air_leaks_i · bar2pa · (P @ u_raw)_i
        F_i = A_c · pressure_i                     (A_c applied internally)
```

matching MATLAB's `F_i = g_i · air_leaks_sf(i) · A_c · bar2pa · p_i`. Gain,
air-leak factor, permutation and `bar2pa` are constant linear factors, so they
commute with the first-order lag and are applied up front. The script **asserts**
both intermediate quantities against the exported `u_model_bar` and `u_force`
rather than trusting the algebra.

The lag stays on **pressure**, upstream of the force map, as in
`simulate_robot_lp.m` — equivalent in force mode, but it means the nonlinear
pressure/`eps0` mode would port unchanged.

### The chamber triad had to be snapped

MATLAB's `Linkage.dc` triad is uniform only to **7.05e-4°** (chamber 1 at
r = 20.000 mm, chambers 2/3 at 20.00043 mm) — a construction artefact in
`robot_linkage.mat`, not round-off from the +15° clock. soromox validates
uniformity to 1e-8 rad and rejects it:

```
ValueError: chamber_azimuth_angles must be uniformly distributed around
the full circle for every pneumatic segment
```

`snap_triad()` replaces it with the exact triad at its circular-mean clock and
**asserts** the deviation under 0.01° rather than bypassing the check. At 7e-4°
on a 20 mm moment arm the actuation error is ~2e-5 relative — microns on a
300 mm swing, three orders below the parity threshold.

### Verified identical, no change needed

Checked against source rather than assumed. `MEG.m:114` and
`pcs.py:_compute_material_damping_full_matrix` / `_local_stiffness_matrix` build
the same tensors:

```
stiffness  L · diag[G·Ix, E·Iy, E·Iz, E·A, G·A, G·A]
damping    L · Eta · diag[Ix, 3·Iy, 3·Iz, 3·A, A, A]
inertia    Rho · diag[Ix, Iy, Iz, A, A, A]
```

with the same 3-chamber annulus cross-section and parallel-axis moments.

**No forward-kinematics tip bug here.** The MATLAB defect — `FwdKinematics`
records `g_here*gi` for a rigid link and only advances by `gf` afterwards,
leaving the last signature 3 mm proximal of the physical tip — has no analogue:
`_forward_kinematics(q, s)` integrates the Magnus expansion to arbitrary arc
length, so `s = sum(L)` lands on the distal end by construction.

---

## Known limitation, reproduced identically on both sides

The physical robot has a resonance at **1.408 Hz, Q ≈ 4.3**. The identified
model has no resonant peak in the swept band and completes zero free-decay
cycles at the identified `Eta`. This is a property of the model, not of either
implementation, and both engines reproduce it to the same digits:

```
per-cycle tip amplitude [mm], square window
meas     200  279  310  304  284  260  245  252  259  266  267  261  255  257
MATLAB   255  251  257  257  258  258  257  258  258  257  258  258  258  258
soromox  255  251  257  257  258  258  257  258  258  257  258  258  258  258
```

The measured envelope rings up and settles by cycle ~6; both simulations are
flat from cycle 1, and both lose 81.6 % of the 5th harmonic. Because the two
agree *with each other* on the miss, it is the missing resonance — not a port
defect.

---

## Parity ladder

Rung 0 runs on every invocation (`--ladder` runs it alone). Rungs 1–3 were not
needed: rung 0 was exact and the full rollout passed on the first attempt.

| # | test | result |
|---|---|---|
| 0 | rest pose `q = 0`, FK vs MATLAB | **0.000000e+00 mm** on tip and mid |
| 1 | unactuated gravity sag | not needed |
| 2 | single-chamber static step | not needed |
| 3 | damped step with the lag | not needed |
| 4 | full rollout, square | **PASS**, 0.0276 mm RMS |

---

## Files

| file | role |
|---|---|
| `examples/simulation/pcs/simulate_isupport_exp.py` | the port; every parameter loaded from the export |
| `examples/simulation/pcs/amisupport_metrics.py` | port of MATLAB `functions/tip_metrics.m`, plus parity statistics |
| `amisupport_dataset/results/PORT_REPORT.md` | full write-up |
| `amisupport_dataset/results/square_*.png`, `square_tip_positions.csv` | figures and per-sample traces |

On the MATLAB side, `python_exporter.m` was rewritten to be batch-safe and
parameterised, and now exports both bags with a parity block (parameters,
linkage geometry read back off the model, and the MATLAB reference rollout).

## Reproducing

```bash
matlab -sd <am_isupportGVS> -batch "python_exporter"

wsl -d Ubuntu-24.04 -- bash -lc '
source ~/soromox/.venv/bin/activate
cd ~/soromox/examples/simulation/pcs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -u simulate_isupport_exp.py --bag square
'
```

Flags: `--ladder` (rung 0 only, seconds), `--tmax` (truncate window, for
timing), `--rtol/--atol` (default to MATLAB's own `ode15s` settings from the
export), `--solver kvaerno5`, `--no-figs`.

## Open items

1. **Interface lengths.** `robot_params.md` documents the short and long
   interfaces as 5 mm and 20 mm; `updateAmISupport.m` — which every
   identification run actually used — builds them at 6 mm and 21 mm. The port
   follows the model (6/21), since `E`, `Eta` and the gains were all fitted
   against it. To be checked against the build.
2. **"Radius 60 mm" in `robot_params.md` is a diameter** (Arleo et al. describe
   a 60 mm-diameter module; the build uses r = 30 mm). Wording only.
3. Chirp bag (`2026-06-08-11-59-18`, 15.22–60.22 s) rollout pending. The MATLAB
   reference scores 244.683 mm there; the chirp is known not to transfer
   (`REPORT.md` §7) — correlation 0.90–0.98 for the first 30 s, collapsing where
   the real arm resonates.
