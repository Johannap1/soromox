"""Why does the simulated tip y coordinate look delayed against VICON?

Reads the per-sample traces written by `simulate_isupport_exp.py` and asks, per
axis: how much of the error is a pure time shift, at what frequency does the
axis actually carry its energy, and does the same shift appear on x and z.

No rollout, no model -- pure signal analysis of the saved CSV.

    python investigate_y_delay.py --bag square
"""

from __future__ import annotations

import argparse
import os

import numpy as np

AXES = ["x", "y", "z"]


def xcorr_lag_subsample(a: np.ndarray, b: np.ndarray, dt: float, max_lag_s: float):
    """Lag of `a` behind `b` in seconds, by normalised cross-correlation.

    Parabolic interpolation around the peak gives sub-sample resolution.
    Returns (lag [s], peak correlation).
    """
    a = a - a.mean()
    b = b - b.mean()
    if a.std() < 1e-15 or b.std() < 1e-15:
        return np.nan, np.nan
    n = a.size
    max_lag = int(min(round(max_lag_s / dt), n - 2))
    lags = np.arange(-max_lag, max_lag + 1)
    c = np.empty(lags.size)
    for i, L in enumerate(lags):
        if L >= 0:
            x, y = a[L:], b[: n - L]
        else:
            x, y = a[: n + L], b[-L:]
        c[i] = (x @ y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-300)
    k = int(np.argmax(c))
    lag = float(lags[k])
    if 0 < k < c.size - 1:  # parabolic refinement
        d = c[k - 1] - 2 * c[k] + c[k + 1]
        if abs(d) > 1e-300:
            lag += 0.5 * (c[k - 1] - c[k + 1]) / d
    return lag * dt, float(c[k])


def band_phase(sig: np.ndarray, t: np.ndarray, f: float):
    """Amplitude and phase of `sig` at frequency `f`, by direct projection."""
    w = 2 * np.pi * f * t
    s = sig - sig.mean()
    c = 2.0 * np.mean(s * np.cos(w))
    q = -2.0 * np.mean(s * np.sin(w))
    return float(np.hypot(c, q)), float(np.degrees(np.arctan2(q, c)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="square")
    ap.add_argument("--outdir", default="amisupport_dataset/results")
    args = ap.parse_args()

    csv_path = os.path.join(args.outdir, f"{args.bag}_tip_positions.csv")
    arr = np.genfromtxt(csv_path, delimiter=",", names=True)
    t = arr["time"]
    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt
    sim = np.vstack([arr["sx_x"], arr["sx_y"], arr["sx_z"]])
    mat = np.vstack([arr["mat_x"], arr["mat_y"], arr["mat_z"]])
    vic = np.vstack([arr["vicon_x"], arr["vicon_y"], arr["vicon_z"]])
    f0 = 1.0

    print(f"source: {csv_path}")
    print(f"{t.size} samples, dt = {dt * 1e3:.3f} ms, fs = {fs:.1f} Hz, "
          f"window {t[0]:.2f}-{t[-1]:.2f} s\n")

    # ---- 1. travel and error per axis ------------------------------------
    print("=" * 78)
    print("1. How much does each axis actually move, and how big is its error?")
    print("=" * 78)
    print(f"{'axis':>5} {'meas RMS':>10} {'meas p-p':>10} {'sim RMS':>10} "
          f"{'RMSE':>9} {'RMSE/meas':>10}")
    for i, ax in enumerate(AXES):
        m = vic[i] - vic[i].mean()
        s = sim[i] - sim[i].mean()
        rmse = float(np.sqrt(np.mean((sim[i] - vic[i]) ** 2)))
        print(f"{ax:>5} {1e3 * m.std():>10.2f} {1e3 * np.ptp(vic[i]):>10.2f} "
              f"{1e3 * s.std():>10.2f} {1e3 * rmse:>9.2f} "
              f"{rmse / max(m.std(), 1e-12):>10.2f}")

    # ---- 2. is the error a pure time shift? ------------------------------
    print("\n" + "=" * 78)
    print("2. Best-fit time shift of SIM behind VICON (>0 = simulation late)")
    print("=" * 78)
    print(f"{'axis':>5} {'lag [ms]':>10} {'lag [deg@1Hz]':>14} {'peak corr':>10} "
          f"{'corr@0 lag':>11} {'RMSE if shifted':>16}")
    for i, ax in enumerate(AXES):
        lag, cpk = xcorr_lag_subsample(sim[i], vic[i], dt, max_lag_s=0.5)
        a = sim[i] - sim[i].mean()
        b = vic[i] - vic[i].mean()
        c0 = float((a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))
        # RMSE after removing the best-fit shift, to see how much it explains
        shift = int(round(lag / dt))
        if shift > 0:
            rs = float(np.sqrt(np.mean((sim[i, shift:] - vic[i, : -shift or None]) ** 2)))
        elif shift < 0:
            rs = float(np.sqrt(np.mean((sim[i, :shift] - vic[i, -shift:]) ** 2)))
        else:
            rs = float(np.sqrt(np.mean((sim[i] - vic[i]) ** 2)))
        print(f"{ax:>5} {1e3 * lag:>10.1f} {360 * f0 * lag:>14.1f} {cpk:>10.3f} "
              f"{c0:>11.3f} {1e3 * rs:>16.2f}")

    # ---- 3. where is each axis' energy? ----------------------------------
    print("\n" + "=" * 78)
    print("3. Spectral content per axis (fraction of variance in each band)")
    print("=" * 78)
    freqs = f0 * np.array([1, 2, 3, 4, 5])
    for i, ax in enumerate(AXES):
        m = vic[i] - vic[i].mean()
        s = sim[i] - sim[i].mean()
        tot_m = float(np.mean(m**2))
        tot_s = float(np.mean(s**2))
        print(f"\n  axis {ax}: measured variance {1e6 * tot_m:.1f} mm^2, "
              f"simulated {1e6 * tot_s:.1f} mm^2")
        print(f"  {'f [Hz]':>7} {'|meas| [mm]':>12} {'|sim| [mm]':>11} "
              f"{'phase meas':>11} {'phase sim':>10} {'dphase':>8} {'dt [ms]':>9}")
        for f in freqs:
            am, pm = band_phase(vic[i], t, f)
            asim, ps = band_phase(sim[i], t, f)
            dph = ((ps - pm + 180) % 360) - 180
            print(f"  {f:>7.2f} {1e3 * am:>12.3f} {1e3 * asim:>11.3f} "
                  f"{pm:>11.1f} {ps:>10.1f} {dph:>8.1f} "
                  f"{1e3 * dph / (360 * f):>9.1f}")
        em = sum(band_phase(vic[i], t, f)[0] ** 2 / 2 for f in freqs)
        print(f"  measured energy captured by 1-5 Hz harmonics: "
              f"{100 * em / max(tot_m, 1e-30):.1f} %")

    # ---- 4. does MATLAB show the same thing? -----------------------------
    print("\n" + "=" * 78)
    print("4. Same lag computed against the MATLAB rollout (port sanity check)")
    print("=" * 78)
    print(f"{'axis':>5} {'sim vs MATLAB lag [ms]':>24} {'peak corr':>10}")
    for i, ax in enumerate(AXES):
        lag, cpk = xcorr_lag_subsample(sim[i], mat[i], dt, max_lag_s=0.5)
        print(f"{ax:>5} {1e3 * lag:>24.3f} {cpk:>10.6f}")

    # ---- 5. bending-plane geometry ---------------------------------------
    print("\n" + "=" * 78)
    print("5. Bending plane: is y a driven axis or the out-of-plane residue?")
    print("=" * 78)
    for name, P in (("VICON", vic), ("soromox", sim)):
        Q = P - P.mean(axis=1, keepdims=True)
        U, S, _ = np.linalg.svd(Q, full_matrices=False)
        frac = S**2 / np.sum(S**2)
        print(f"  {name:>8}: principal dirs")
        for k in range(3):
            print(f"      pc{k + 1} [{U[0, k]:+.3f} {U[1, k]:+.3f} {U[2, k]:+.3f}] "
                  f"  {100 * frac[k]:5.1f} % of variance")
    n_v = np.linalg.svd(vic - vic.mean(axis=1, keepdims=True), full_matrices=False)[0][:, 2]
    n_s = np.linalg.svd(sim - sim.mean(axis=1, keepdims=True), full_matrices=False)[0][:, 2]
    ang = np.degrees(np.arccos(abs(float(n_v @ n_s))))
    print(f"\n  angle between measured and simulated bending planes: {ang:.2f} deg")

    # How much y does a small plane misalignment leak from the main motion?
    z1, _ = band_phase(vic[2], t, f0)
    print(f"  1 Hz z amplitude {1e3 * z1:.1f} mm x sin({ang:.2f} deg) = "
          f"{1e3 * z1 * np.sin(np.radians(ang)):.1f} mm of apparent y")

    make_figure(t, sim, vic, dt, args.outdir, args.bag)


def make_figure(t, sim, vic, dt, outdir, bag):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"figure.dpi": 120, "savefig.dpi": 200, "font.size": 10,
         "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
         "axes.spines.right": False, "legend.frameon": False}
    )
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))

    # (a) the y trace as it looks -- "delayed"
    axes[0].plot(t, 1e3 * vic[1], color="#3f3f3f", lw=1.4, alpha=0.7, label="VICON y")
    axes[0].plot(t, 1e3 * sim[1], color="#d1495b", lw=1.4, label="soromox y")
    axes[0].set_ylabel("tip y [mm]")
    axes[0].set_xlim(t[0], min(t[-1], t[0] + 6))
    axes[0].legend(ncol=2, loc="upper right")
    axes[0].set_title("(a) y looks shifted -- but the simulated swing is 4.4x too small")

    # (b) the same, with the simulation SIGN-FLIPPED and rescaled
    k = float(np.std(vic[1]) / max(np.std(sim[1]), 1e-15))
    axes[1].plot(t, 1e3 * (vic[1] - vic[1].mean()), color="#3f3f3f", lw=1.4,
                 alpha=0.7, label="VICON y (demeaned)")
    axes[1].plot(t, -k * 1e3 * (sim[1] - sim[1].mean()), color="#1f4e79", lw=1.4,
                 label=f"soromox y, negated and scaled x{k:.1f}")
    axes[1].set_ylabel("tip y [mm]")
    axes[1].set_xlim(t[0], min(t[-1], t[0] + 6))
    axes[1].legend(ncol=2, loc="upper right")
    axes[1].set_title("(b) negating it aligns the phase: the offset is ~180 deg, not a lag")

    # (c) cross-correlation vs lag, per axis
    n = t.size
    max_lag = int(round(0.6 / dt))
    lags = np.arange(-max_lag, max_lag + 1)
    for i, (ax_name, col) in enumerate(zip(AXES, ["#1f4e79", "#d1495b", "#2a9d8f"])):
        a = sim[i] - sim[i].mean()
        b = vic[i] - vic[i].mean()
        c = np.empty(lags.size)
        for j, L in enumerate(lags):
            if L >= 0:
                x, y = a[L:], b[: n - L]
            else:
                x, y = a[: n + L], b[-L:]
            c[j] = (x @ y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-300)
        axes[2].plot(1e3 * lags * dt, c, color=col, lw=1.5, label=f"{ax_name}")
    axes[2].axvline(0, color="k", lw=0.8)
    axes[2].axhline(0, color="k", lw=0.6)
    axes[2].set_xlabel("lag of simulation behind VICON [ms]")
    axes[2].set_ylabel("normalised correlation")
    axes[2].legend(ncol=3, loc="lower right")
    axes[2].set_title(
        "(c) x and z peak at zero lag; y has no peak near zero -- it is anti-correlated"
    )

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{bag}_y_delay_diagnosis.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    main()
