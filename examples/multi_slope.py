"""Multi-slope amplitude recovery (reproduces bai/multi_slope_amp.m).

Synthetic decaying noise with three slopes:
    A = [-20, -20, -40] dB,  T = [1, 2, inf] s,  fs = 2000 Hz,  2 s.
The third slope (T = inf) is the stationary noise floor. The LINEX estimator
recovers all three amplitudes with ~zero bias and 100% convergence. Slopes 1 and
2 (similar amplitude, nearby decay times) are correlated, so each has larger
per-realization variance while their mean stays on target -- as in the paper.

Run:  python examples/multi_slope.py [N]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from multi_slope_linex import generate_decaying_noise, common_slope_fit, decay_kernel


def main(n=500):
    amps = [-20.0, -20.0, -40.0]
    Tlist = [1.0, 2.0, np.inf]
    fs, dur = 2000, 2.0
    labels = [r"$A_1$ (T=1s)", r"$A_2$ (T=2s)", r"$A_0$ (noise)"]

    h, t, _, _ = generate_decaying_noise(amps, Tlist, fs=fs, duration_s=dur, n=n, seed=1)
    a_vals, infos = common_slope_fit((h ** 2).T, Tlist, fs=fs)
    rec = 10.0 * np.log10(np.maximum(a_vals, 1e-300))
    conv = 100.0 * np.mean([i["converged"] for i in infos])

    print(f"Multi-slope LINEX recovery  (truth = {amps} dB, T = {Tlist}, fs = {fs}, N = {n})")
    for j, lab in enumerate(["T=1s ", "T=2s ", "noise"]):
        print(f"  slope {lab}: mean = {rec[:, j].mean():+.3f} dB  "
              f"bias = {rec[:, j].mean() - amps[j]:+.3f}  std = {rec[:, j].std():.3f}")
    print(f"  convergence = {conv:.0f}%")

    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.35, wspace=0.3)

    # Top row: recovered-amplitude distributions per slope.
    for j in range(3):
        ax = fig.add_subplot(gs[0, j])
        col = rec[:, j]
        ax.hist(col, bins=40, density=True, alpha=0.35, color=f"C{j}")
        if col.std() > 1e-9:
            xs = np.linspace(col.min() - 0.5, col.max() + 0.5, 400)
            ax.plot(xs, gaussian_kde(col)(xs), color=f"C{j}", lw=2)
        ax.axvline(amps[j], color="r", ls="--", lw=1.2)
        ax.set_title(labels[j])
        ax.set_xlabel("A (dB)")
        if j == 0:
            ax.set_ylabel("pdf")

    # Bottom: one example fit (energy vs. fitted model and per-slope components).
    ax = fig.add_subplot(gs[1, :])
    idx = 0
    X = decay_kernel(Tlist, t)
    w = a_vals[idx]
    energy_db = 10.0 * np.log10(np.maximum(h[:, idx] ** 2, 1e-300))
    fit_db = 10.0 * np.log10(np.maximum(X @ w, 1e-300))
    ax.plot(t, energy_db, color="0.8", lw=0.7, label="squared signal $h^2$")
    ax.plot(t, fit_db, color="C3", lw=2, label="fitted model")
    styles = ["--", "-.", ":"]
    for j in range(3):
        comp_db = 10.0 * np.log10(np.maximum(X[:, j] * w[j], 1e-300))
        ax.plot(t, comp_db, color="0.25", lw=1, ls=styles[j], label=f"slope {j + 1}")
    ax.set_xlim(0, dur)
    ax.set_ylim(-80, -10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("energy (dB)")
    ax.set_title(f"Example LINEX fit  (N={n}, convergence={conv:.0f}%)")
    ax.legend(ncol=3, fontsize=8)

    fig.suptitle("LINEX common-slope amplitude estimation", y=0.98)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multi_slope.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  figure saved to {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
