"""Tracking metrics for the AM-I Support forced-oscillation experiments.

Direct port of ``functions/tip_metrics.m`` from the MATLAB GVS repository, so
that MATLAB and soromox rollouts are scored by exactly the same numbers.

The square-wave experiment is a 1 Hz FORCED oscillation: the tip never settles,
so a "resonance-window RMSE" is meaningless. What is reported instead is
(a) how big the swing is, (b) whether it arrives on time, and (c) how the energy
is distributed over the drive harmonics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TipMetrics:
    f0: float
    rmse_full: float
    rmse_axis: np.ndarray
    dir: np.ndarray
    amp_meas: np.ndarray
    amp_sim: np.ndarray
    amp_err: np.ndarray
    amp_err_mean: float
    amp_err_rel: float
    lag_sim: np.ndarray
    lag_meas: np.ndarray
    phase_lag: np.ndarray
    phase_lag_mean: float
    spec_f: np.ndarray
    spec_meas: np.ndarray
    spec_sim: np.ndarray
    spec_err: np.ndarray
    spec_err_rel: np.ndarray
    offset_t0: float
    extra: dict = field(default_factory=dict)


def _xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    """Integer lag (samples) of ``a`` behind ``b`` maximising normalised correlation."""
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    if a.std() < np.finfo(float).eps or b.std() < np.finfo(float).eps:
        return np.nan
    n = a.size
    max_lag = int(min(max_lag, n - 2))
    if max_lag < 1:
        return np.nan
    lags = np.arange(-max_lag, max_lag + 1)
    c = np.zeros(lags.size)
    for i, L in enumerate(lags):
        if L >= 0:
            x, y = a[L:], b[: n - L]
        else:
            x, y = a[: n + L], b[-L:]
        denom = max(np.linalg.norm(x) * np.linalg.norm(y), np.finfo(float).eps)
        c[i] = float(x @ y) / denom
    return float(lags[int(np.argmax(c))])


def tip_metrics(
    t: np.ndarray,
    p_sim: np.ndarray,
    p_meas: np.ndarray,
    u: np.ndarray | None = None,
    f0: float = 1.0,
    harmonics: np.ndarray | None = None,
) -> TipMetrics:
    """Port of ``tip_metrics.m``.

    Args:
        t: (M,) time [s], relative to the window start.
        p_sim: (3, M) simulated tip position [m].
        p_meas: (3, M) measured tip position [m].
        u: (n, M) actuation input, used only as the phase reference.
        f0: drive frequency [Hz].
        harmonics: harmonics to report [Hz]; default ``f0 * [1, 3, 5]``.
    """
    t = np.asarray(t, dtype=float).ravel()
    p_sim = np.asarray(p_sim, dtype=float).reshape(3, -1)
    p_meas = np.asarray(p_meas, dtype=float).reshape(3, -1)

    M = t.size
    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt
    harm = np.asarray(harmonics if harmonics is not None else f0 * np.array([1, 3, 5]))

    # 1. Plain RMSE (3D and per axis) over the full window
    err = p_sim - p_meas
    rmse_full = float(np.sqrt(np.mean(np.sum(err**2, axis=0))))
    rmse_axis = np.sqrt(np.mean(err**2, axis=1))

    # 2. Project onto the dominant MEASURED motion direction. Using the measured
    #    direction (not the simulated one) keeps the projection independent of
    #    the model parameters.
    Pm = p_meas - p_meas.mean(axis=1, keepdims=True)
    U, _, _ = np.linalg.svd(Pm, full_matrices=False)
    direction = U[:, 0]
    if direction @ Pm[:, -1] < 0:
        direction = -direction

    s_meas = direction @ (p_meas - p_meas.mean(axis=1, keepdims=True))
    # Same origin as s_meas, so a static bias stays visible.
    s_sim = direction @ (p_sim - p_meas.mean(axis=1, keepdims=True))

    # 3. Per-cycle amplitude
    ncyc = max(1, int(np.floor((t[-1] - t[0]) * f0)))
    edges = t[0] + np.arange(ncyc + 1) / f0
    amp_meas = np.full(ncyc, np.nan)
    amp_sim = np.full(ncyc, np.nan)
    lag_sim = np.full(ncyc, np.nan)
    lag_meas = np.full(ncyc, np.nan)

    ref = None
    if u is not None and np.size(u) > 0:
        u = np.asarray(u, dtype=float).reshape(-1, M)
        ch = int(np.argmax(u.var(axis=1)))
        ref = u[ch] - u[ch].mean()

    max_lag = int(round(fs / f0 / 2))  # +/- half a drive period

    for k in range(ncyc):
        idx = (t >= edges[k]) & (t < edges[k + 1])
        if idx.sum() < 3:
            continue
        amp_meas[k] = 0.5 * (s_meas[idx].max() - s_meas[idx].min())
        amp_sim[k] = 0.5 * (s_sim[idx].max() - s_sim[idx].min())
        if ref is not None:
            lag_sim[k] = _xcorr_lag(s_sim[idx], ref[idx], max_lag) * dt
            lag_meas[k] = _xcorr_lag(s_meas[idx], ref[idx], max_lag) * dt

    amp_err = amp_sim - amp_meas
    amp_err_mean = float(np.nanmean(np.abs(amp_err)))
    amp_err_rel = amp_err_mean / max(float(np.nanmean(amp_meas)), np.finfo(float).eps)

    phase_lag = 360.0 * f0 * (lag_sim - lag_meas)  # deg, >0 = simulation late

    # 4. Harmonic content. The fit window is an integer number of drive periods
    #    by construction, so the harmonics land on exact FFT bins.
    fgrid = np.arange(M) * fs / M
    S_meas = np.abs(np.fft.fft(s_meas, M)) * 2.0 / M
    S_sim = np.abs(np.fft.fft(s_sim, M)) * 2.0 / M

    nh = harm.size
    spec_f = np.full(nh, np.nan)
    spec_meas = np.full(nh, np.nan)
    spec_sim = np.full(nh, np.nan)
    half = M // 2
    for h in range(nh):
        b = int(np.argmin(np.abs(fgrid[:half] - harm[h])))
        if abs(fgrid[b] - harm[h]) > fs / M:
            continue  # harmonic not resolved
        spec_f[h] = fgrid[b]
        spec_meas[h] = S_meas[b]
        spec_sim[h] = S_sim[b]

    spec_err = spec_sim - spec_meas
    spec_err_rel = spec_err / np.maximum(spec_meas, np.finfo(float).eps)

    return TipMetrics(
        f0=f0,
        rmse_full=rmse_full,
        rmse_axis=rmse_axis,
        dir=direction,
        amp_meas=amp_meas,
        amp_sim=amp_sim,
        amp_err=amp_err,
        amp_err_mean=amp_err_mean,
        amp_err_rel=amp_err_rel,
        lag_sim=lag_sim,
        lag_meas=lag_meas,
        phase_lag=phase_lag,
        phase_lag_mean=float(np.nanmean(phase_lag)),
        spec_f=spec_f,
        spec_meas=spec_meas,
        spec_sim=spec_sim,
        spec_err=spec_err,
        spec_err_rel=spec_err_rel,
        offset_t0=float(np.linalg.norm(p_sim[:, 0] - p_meas[:, 0])),
    )


def print_tip_metrics(m: TipMetrics, label: str = "TIP") -> None:
    """Mirror of ``print_tip_metrics`` in identification.m."""
    print(f"\n===== {label} tracking metrics =====")
    print(f"RMSE (full window, 3D)   : {1e3 * m.rmse_full:8.3f} mm")
    print(
        "RMSE per axis (x,y,z)    : "
        + " ".join(f"{1e3 * v:8.3f}" for v in m.rmse_axis)
        + " mm"
    )
    print(f"Static offset at t0      : {1e3 * m.offset_t0:8.3f} mm  (geometric floor)")
    print("Dominant motion dir      : [" + " ".join(f"{v:6.3f}" for v in m.dir) + "]")
    print("-- per-cycle amplitude (half peak-to-peak along dominant dir) --")
    print("  cycle : " + "".join(f"{i + 1:9d}" for i in range(m.amp_meas.size)))
    print("  meas  : " + "".join(f"{1e3 * v:9.2f}" for v in m.amp_meas) + "  mm")
    print("  sim   : " + "".join(f"{1e3 * v:9.2f}" for v in m.amp_sim) + "  mm")
    print("  err   : " + "".join(f"{1e3 * v:9.2f}" for v in m.amp_err) + "  mm")
    print(
        f"  mean |amp err| = {1e3 * m.amp_err_mean:.3f} mm "
        f"({100 * m.amp_err_rel:.1f} % of mean amplitude)"
    )
    print(f"-- per-cycle phase lag vs input (deg at {m.f0:.2f} Hz, >0 = sim late) --")
    print("  lag   : " + "".join(f"{v:9.1f}" for v in m.phase_lag) + "  deg")
    print(f"  mean phase lag = {m.phase_lag_mean:.2f} deg")
    print("-- tip displacement spectrum --")
    for i in range(m.spec_f.size):
        print(
            f"  f = {m.spec_f[i]:5.2f} Hz : meas {1e3 * m.spec_meas[i]:8.3f} mm | "
            f"sim {1e3 * m.spec_sim[i]:8.3f} mm | "
            f"err {1e3 * m.spec_err[i]:+8.3f} mm ({100 * m.spec_err_rel[i]:+7.1f} %)"
        )
    print("===============================")


def parity_stats(p_a: np.ndarray, p_b: np.ndarray) -> dict:
    """Difference statistics between two SIMULATED trajectories (3, M), in metres.

    ``p_a`` is the reference (MATLAB), ``p_b`` the port (soromox).
    """
    p_a = np.asarray(p_a, dtype=float).reshape(3, -1)
    p_b = np.asarray(p_b, dtype=float).reshape(3, -1)
    d = p_b - p_a
    n = np.linalg.norm(d, axis=0)
    span = p_a - p_a.mean(axis=1, keepdims=True)
    ptp = 2.0 * float(np.max(np.linalg.norm(span, axis=0)))
    rms = float(np.sqrt(np.mean(n**2)))
    return {
        "rms": rms,
        "rms_axis": np.sqrt(np.mean(d**2, axis=1)),
        "max": float(n.max()),
        "argmax": int(np.argmax(n)),
        "mean_signed": d.mean(axis=1),
        "ptp_reference": ptp,
        "rms_rel_ptp": rms / max(ptp, np.finfo(float).eps),
    }


def print_parity(stats: dict, label: str, threshold_mm: float = 1.0) -> bool:
    """Print a parity block; return True when it passes ``threshold_mm``."""
    ok = 1e3 * stats["rms"] <= threshold_mm
    print(f"\n===== PARITY ({label}) : soromox vs MATLAB =====")
    print(f"RMS difference        : {1e3 * stats['rms']:8.4f} mm")
    print(
        "  per axis (x,y,z)    : "
        + " ".join(f"{1e3 * v:8.4f}" for v in stats["rms_axis"])
        + " mm"
    )
    print(
        f"max difference        : {1e3 * stats['max']:8.4f} mm "
        f"(sample {stats['argmax']})"
    )
    print(
        "mean signed offset    : "
        + " ".join(f"{1e3 * v:+8.4f}" for v in stats["mean_signed"])
        + " mm"
    )
    print(f"reference peak-to-peak: {1e3 * stats['ptp_reference']:8.1f} mm")
    print(f"RMS / peak-to-peak    : {100 * stats['rms_rel_ptp']:8.4f} %")
    print(f"VERDICT               : {'PASS' if ok else 'FAIL'} (threshold {threshold_mm:.1f} mm)")
    print("=" * 46)
    return ok
