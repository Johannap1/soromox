# AM-I Support: porting the identified GVS model from MATLAB/SoRoSim to soromox

Port of the identified AM-I Support model from the MATLAB GVS pipeline
(`am_isupportGVS`, branch `identification-geometry-fixes`) to soromox's
JAX/diffrax PCS implementation, and verification that the two produce the same
trajectory.

**Headline: the two simulations agree to 0.028 mm RMS on a 625 mm peak-to-peak
tip swing (0.0044 %), 36x inside the 1 mm acceptance threshold, and soromox
reproduces the MATLAB tip RMSE of 40.128 mm against VICON to three decimals.**

Source of truth: `results/identification_T1_fc28_0d2ecd93.mat` (run `T1_fc28`,
config hash `0d2ecd93`). See `REPORT.md` on the MATLAB side for how those
parameters were obtained.

---

## 1. Result

### Square (bag `2026-06-08-12-11-25`, 27.60-42.60 s)

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

Against VICON, every reported metric matches:

| metric | MATLAB | soromox |
|---|---|---|
| tip RMSE (3D) | 40.128 mm | 40.128 mm |
| tip RMSE per axis (x, y, z) | 20.089 / 14.486 / 31.573 mm | 20.089 / 14.486 / 31.572 mm |
| static offset at t0 | 2.918 mm | 2.918 mm |
| mean amplitude error | 18.270 mm (6.9 %) | 18.269 mm (6.9 %) |
| mean phase lag | -3.60 deg | -3.60 deg |
| spectrum 1 / 3 / 5 Hz | -0.7 % / +14.3 % / -81.6 % | -0.7 % / +14.3 % / -81.6 % |

These reproduce the `T1_fc28` row of the MATLAB `REPORT.md` line for line.

The square result is **insensitive to solver tolerance**. Repeating it at
MATLAB's own `rtol 1e-3 / atol 1e-4` reproduced every figure above to four
decimals (see §2).

### Chirp (bag `2026-06-08-11-59-18`, 15.22-60.22 s)

**Not completed.** The rollout ran ~7 h without finishing and was stopped
deliberately; see §6 for why it is so expensive. The MATLAB reference for this
window is exported and scores 244.683 mm against VICON
(x 138.74 | y 138.34 | z 146.56), so the comparison can be finished at any time
by re-running `--bag chirp`.

The chirp is known not to transfer (`REPORT.md` §7): correlation is 0.90-0.98
for the first 30 s and collapses only where the real arm resonates. Nothing
about the port depends on it -- it would test parity on a second, harder
trajectory, not establish it.

---

## 2. The residual 0.028 mm is solver noise, not a model difference

The two engines integrate different ODE solvers at different tolerances:

| | MATLAB | soromox |
|---|---|---|
| solver | `ode15s` (implicit, variable order) | `Tsit5` (explicit RK) |
| step control | `RelTol 1e-3`, `AbsTol 1e-4` | `PIDController(rtol=1e-5, atol=1e-7)` |

The difference grows mildly across the window and carries no structure tied to
the 1 Hz drive -- the signature of independent step-size histories rather than a
constitutive or geometric discrepancy. A 1 s probe of the same window scored
0.0084 mm; the full 15 s window scores 0.0276 mm.

Rerunning the square at MATLAB's own tolerances changes **nothing**:

| | rtol 1e-5 / atol 1e-7 | rtol 1e-3 / atol 1e-4 |
|---|---|---|
| wall time | 11 632 s | 11 374 s |
| parity RMS | 0.0276 mm | 0.0276 mm |
| per axis | 0.0115 / 0.0060 / 0.0244 | 0.0115 / 0.0060 / 0.0244 |
| max difference | 0.0896 mm @ 519 | 0.0896 mm @ 519 |
| vs VICON | 40.128 mm | 40.128 mm |

Identical to four decimals at 100x looser tolerance. That is itself the
diagnosis of the runtime problem -- see §6.

---

## 3. What had to be reconciled

### 3.1 Geometry: MATLAB link chain -> soromox PCS layout

The MATLAB linkage is `r s r r s r`. soromox wants alternating rigid/soft
segments, so the two consecutive rigid links in the middle (6 mm short interface
+ 21 mm long interface) merge into one 27 mm body:

| seg | kind | L [mm] | r [mm] | rho [kg/m3] |
|---|---|---|---|---|
| 0 | rigid | 42.000 | 30.000 | 1210.0 |
| 1 | soft | 180.000 | 28.600 | 2721.2 |
| 2 | rigid | 27.000 | 30.000 | 1210.0 |
| 3 | soft | 180.000 | 28.600 | 2721.2 |
| 4 | rigid | 6.000 | 30.000 | 1210.0 |

**The base segment is 42 mm, not 41 mm.** `g_ini` carries a 36 mm translation
along the base-frame x axis on top of the 6 mm short interface. Folding it into
segment 0 is exact rather than approximate: `g_ini`'s rotation is a roll about
x, so it leaves the x axis invariant and the offset is collinear with the first
link. The extra mass this gives segment 0 has no dynamic effect -- the segment is
proximal to every strain DOF, so nothing can move it.

Three values in the previous `simulate_isupport_exp.py` were wrong against the
identified model and are now read from the export instead of hardcoded:

| | was | is |
|---|---|---|
| base segment length | 41 mm | 42 mm |
| soft segment radius | 30 mm | 28.6 mm |
| soft segment density | 6471 kg/m3 | 2721.2 kg/m3 |

### 3.2 Constitutive model: already identical

Verified against source rather than assumed. `MEG.m:114` and
`pcs.py:_compute_material_damping_full_matrix` / `_local_stiffness_matrix` build
the same tensors:

```
stiffness  L * diag[G*Ix, E*Iy, E*Iz, E*A, G*A, G*A]
damping    L * Eta * diag[Ix, 3*Iy, 3*Iz, 3*A, A, A]
inertia    Rho * diag[Ix, Iy, Iz, A, A, A]
```

with the same 3-chamber annulus cross-section: `A = 3*pi*(ro^2 - ri^2)`,
`I0 = pi/4*(ro^4 - ri^4)` per chamber, parallel-axis about `delta` at the
chamber azimuths, `Ix = Iy + Iz`. No changes were needed on either side.

Note that MATLAB's `MEG.m` uses `VLinks(2).theta` (the *unclocked* 90/210/330
triad) for these moments while actuation uses the clocked `dc`. This is not an
inconsistency that matters: for a 3-fold symmetric triad, `sum sin^2(theta + k*120)`
is 3/2 for any clock, so `Iy` and `Iz` are clock-invariant.

### 3.3 Forward kinematics: no body-frame bug in soromox

The MATLAB bug (`FwdKinematics` records `g_here*gi` for a rigid link and only
advances by `gf` afterwards, leaving the last signature 3 mm proximal of the
physical tip) has no analogue here. `_forward_kinematics(q, s)` integrates the
Magnus expansion to arbitrary arc length, so evaluating at `s = sum(L)` lands on
the distal end by construction.

Arc lengths come from the export, not from re-deriving the signature indexing:
at `q = 0` the arm is straight along the base x axis, so the x coordinate of any
frame *is* its arc length. That gives `s_tip = 435.0000 mm`,
`s_mid = 249.0000 mm`.

### 3.4 Actuation

MATLAB (force mode) and soromox reach the same generalized force by different
routes:

```
MATLAB   u_force = air_leaks_sf * A_c * bar2pa * (P @ u_raw)      [N], pre-gain
         F_i     = g_i * u_force_i                (gain applied after the lag)

soromox  pressure fed to the solver = g_i * air_leaks_i * bar2pa * (P @ u_raw)_i
         F_i = A_c * pressure_i                   (A_c applied internally)
```

Gain, air-leak factor, permutation and `bar2pa` are all constant linear factors,
so they commute with the first-order lag and may be applied up front. The script
asserts both intermediate quantities against the exported `u_model_bar` and
`u_force` rather than trusting the algebra.

The lag stays on **pressure**, upstream of the force map, as in
`simulate_robot_lp.m`. In force mode this is equivalent to filtering the force,
but keeping the ordering means the pressure/`eps0` mode would port unchanged.

### 3.5 The chamber triad had to be snapped

MATLAB's `Linkage.dc` triad is uniform only to **7.05e-4 deg**, with chamber 1 at
radius 20.000 mm and chambers 2/3 at 20.00043 mm -- a construction artefact baked
into `robot_linkage.mat`, not round-off from the +15 deg clock rotation. soromox
validates uniformity to 1e-8 rad and rejects anything looser:

```
ValueError: chamber_azimuth_angles must be uniformly distributed around
the full circle for every pneumatic segment
```

The triad is replaced by the exact one at its circular-mean clock, and the
deviation is **asserted** (`snap_triad`, tolerance 0.01 deg) rather than
bypassed. At 7e-4 deg on a 20 mm moment arm the actuation error is ~2e-5
relative -- microns on a 300 mm swing, three orders below the parity threshold.
soromox also carries one `chamber_distance` per pneumatic segment, so the 0.4 um
radius spread cannot be represented; the mean is used and the spread asserted
below 10 um.

---

## 4. Parity ladder

Rung 0 is run on every invocation (`--ladder` runs it alone). Rungs 1-3 were not
needed: rung 0 was exact and the full rollout passed on the first attempt.

| # | test | result |
|---|---|---|
| 0 | rest pose, `q = 0`, FK vs MATLAB | **0.000000e+00 mm** on both tip and mid |
| 1 | unactuated gravity sag | not needed |
| 2 | single-chamber static step | not needed |
| 3 | damped step with the lag | not needed |
| 4 | full rollout, square | **PASS**, 0.0276 mm RMS |
| 4 | full rollout, chirp | pending |

---

## 5. Known limitation, reproduced identically on both sides

The physical robot has a resonance at **1.408 Hz, Q ~ 4.3**. The model has no
resonant peak in the swept band and completes zero free-decay cycles at the
identified `Eta`. This is a property of the identified model, not of either
implementation, and both engines reproduce it to the same digits:

```
per-cycle tip amplitude [mm], square window
meas     200  279  310  304  284  260  245  252  259  266  267  261  255  257
MATLAB   255  251  257  257  258  258  257  258  258  257  258  258  258  258
soromox  255  251  257  257  258  258  257  258  258  257  258  258  258  258
```

The measured envelope rings up and settles by cycle ~6; both simulations are
flat from cycle 1. Both lose 81.6 % of the 5th harmonic. Because the two agree
*with each other* on the miss, it is the missing resonance and not a port
defect.

---

## 5b. The simulated tip y is inverted, not delayed

The simulated `y` looks time-shifted against VICON. It is not: cross-correlation
reports -492 ms, which at 1 Hz is **-177.2 deg** -- half a period, and on a
periodic signal a half-period shift is indistinguishable from a sign flip. The
direct 1 Hz phase difference agrees at -169.2 deg, and the correlation at *zero*
lag is **negative** (-0.130). Negating the simulated y aligns it.

```
axis   lag [ms]  lag [deg@1Hz]  peak corr  corr@0 lag
   x        0.7            0.2      0.915       0.915
   y     -492.2         -177.2      0.589      -0.130
   z       -4.9           -1.8      0.985       0.985
```

x and z show no lag, so there is no timing error anywhere in the model. It is
also not a port artefact: soromox vs MATLAB on y gives -0.046 ms with
correlation 0.999999.

The cause is that y is the **out-of-plane axis**. It carries 0.4 % of the
measured variance, and only **22 %** of its energy sits in the 1-5 Hz drive
harmonics (x: 83 %, z: 95 %). The model moves 4.4x too little there (3.10 vs
13.64 mm RMS). What little 1 Hz y exists on either side is leakage from the main
bending motion through a residual **2.74 deg** plane misalignment:
`248.2 mm x sin(2.74 deg) = 11.9 mm`, the right order for the measured 8.5 mm.
The sign of that leakage sets the apparent inversion.

This is the floor `REPORT.md` §5 already identified -- a constant-y model scores
13.64 mm and we score 14.49 mm. Closing it means removing the last 2.74 deg of
plane misalignment, not fixing timing.

Reproduce with `python investigate_y_delay.py --bag square`
(figure: `square_y_delay_diagnosis.png`).

---

## 6. Runtime: stiffness, and why there is no cheap fix

| run | wall time |
|---|---|
| rung 0 (`--ladder`) | seconds |
| 1 s window, Tsit5 | 706 s |
| square 15 s window, Tsit5 | 11 374 s (3 h 09 m) |
| chirp 45 s window, Tsit5 | not completed, stopped after ~7 h |
| **MATLAB `ode15s`, same 15 s window** | **28 s** |

The model is stiff: `Eta = 3.67e5 Pa*s` leaves it overdamped by roughly an order
of magnitude (`REPORT.md` §6 -- zero free-decay cycles). For an **explicit**
method the step size is then capped by *stability*, which is independent of the
accuracy target. That is exactly what the tolerance experiment in §2 shows: 100x
looser tolerance, same runtime, same answer to four decimals.

The obvious remedy -- an implicit solver -- was tried and is **worse**.
`Kvaerno5` failed to finish a **1 second** window in 50 min, against Tsit5's
11 m 46 s, because every implicit step costs a Jacobian of the full PCS dynamics
plus Newton iterations, and here that per-step cost swamps the step-count
saving. MATLAB's `ode15s` wins on both counts only because SoRoSim hands it a
much cheaper right-hand side to differentiate.

So neither tolerance nor solver choice helps. A real fix means work inside
soromox -- an analytic or sparsity-exploiting Jacobian, or a cheaper implicit
scheme -- which is a change to its integrator/dynamics and out of scope here.

**Practical consequence: validate on the square.** It costs ~3 h; the chirp
costs most of a day and mainly re-confirms a known result.

---

## 7. Reproducing

MATLAB side (export both bags, including the reference rollouts):

```bash
matlab -sd <am_isupportGVS> -batch "python_exporter"
```

Writes `dataset/python_exported/<bagname>_py.mat` with the VICON poses, the raw
actuation, the identified parameters, the linkage geometry read back off the
model, and the MATLAB reference rollout on the fit window. If
`results/identification_T1_fc28_*.mat` is absent (results are gitignored),
regenerate it first:

```bash
matlab -sd <am_isupportGVS> -batch "setenv('AMI_RUN','T1_fc28'); identification"
```

soromox side:

```bash
wsl -d Ubuntu-24.04 -- bash -lc '
source ~/soromox/.venv/bin/activate
cd ~/soromox/examples/simulation/pcs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -u simulate_isupport_exp.py --bag both
'
```

Useful flags: `--ladder` (rung 0 only, seconds), `--tmax 1.0` (truncate the
window, for timing), `--rtol/--atol` (solver tolerance), `--solver kvaerno5`
(implicit, if the explicit solver ever stalls on stiffness), `--no-figs`.

There are **no model literals** in `simulate_isupport_exp.py`. Re-identify on
the MATLAB side, re-export, re-run, and the two stay in step.

Costs are tabulated in §6. Budget ~3 h for the square and do not expect
tolerance or solver changes to help.

---

## 8. Files

| file | role |
|---|---|
| `examples/simulation/pcs/simulate_isupport_exp.py` | the port; loads every parameter from the export |
| `examples/simulation/pcs/amisupport_metrics.py` | port of MATLAB `functions/tip_metrics.m`, plus parity statistics |
| `<am_isupportGVS>/python_exporter.m` | batch-safe, parameterised export of both bags |
| `amisupport_dataset/results/square_tip_traces.png` | MATLAB vs soromox vs VICON, per axis |
| `amisupport_dataset/results/square_axis_error.png` | per-axis error against VICON, both engines |
| `amisupport_dataset/results/square_parity_residual.png` | soromox - MATLAB, with the threshold |
| `amisupport_dataset/results/square_tip_3d.png` | trajectory overlay |
| `amisupport_dataset/results/square_tip_positions.csv` | per-sample traces |
| `examples/simulation/pcs/investigate_y_delay.py` | the y-axis analysis of §5b |
| `amisupport_dataset/results/square_y_delay_diagnosis.png` | its figure |
| `amisupport_dataset/results/PR_BODY.md` | pull-request description |

---

## 9. Open items

0. **Chirp rollout not run to completion** (stopped after ~7 h). Re-run with
   `python -u simulate_isupport_exp.py --bag chirp` -- the MATLAB reference is
   already in the export, so nothing on the MATLAB side needs redoing.
1. **Interface lengths.** `robot_params.md` documents the short and long
   interfaces as 5 mm and 20 mm; `updateAmISupport.m` -- which every
   identification run actually used -- builds them at 6 mm and 21 mm. The port
   follows the model (6/21), since `E`, `Eta` and the gains were all fitted
   against it. Deferred, to be checked against the build.
2. **"Radius 60 mm" in `robot_params.md` is a diameter.** Arleo et al. describe
   a 60 mm-diameter module and the build uses r = 30 mm. Documentation wording
   only.
3. **soromox branch `am_isupport` is unverified against origin.** `git fetch`
   hangs on the SSH remote with no agent available in this shell, and the branch
   has no upstream tracking ref. The local tree is clean at `7ab0d40`.
4. Work discarded from `dynamic_identification` at the start of the port is
   recoverable as `stash@{0}: On dynamic_identification: pre-port-discard`.
