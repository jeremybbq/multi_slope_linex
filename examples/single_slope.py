"""Single-slope amplitude recovery (reproduces bai/single_slope_amp.m).

Synthetic decaying noise: A = -10 dB, T = 0.8 s, fs = 250 Hz, 0.6 s.
The LINEX estimator should recover -10 dB with ~zero bias and 100% convergence.

Run:  python examples/single_slope.py [N]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from multi_slope_linex import generate_decaying_noise, common_slope_fit


def main(n=500):
    amp_db, T, fs, dur = -10.0, 0.8, 250, 0.6
    h, t, _, _ = generate_decaying_noise([amp_db], [T], fs=fs, duration_s=dur, n=n, seed=1)

    a_vals, infos = common_slope_fit((h ** 2).T, [T], fs=fs)
    rec = 10.0 * np.log10(a_vals[:, 0])
    conv = 100.0 * np.mean([i["converged"] for i in infos])

    print(f"Single-slope LINEX recovery  (truth = {amp_db:.1f} dB, T = {T}s, fs = {fs}, N = {n})")
    print(f"  mean   = {rec.mean():+.3f} dB   (bias = {rec.mean() - amp_db:+.3f} dB)")
    print(f"  median = {np.median(rec):+.3f} dB   std = {rec.std():.3f} dB")
    print(f"  convergence = {conv:.0f}%")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rec, bins=40, density=True, alpha=0.35, color="C0")
    xs = np.linspace(rec.min() - 0.5, rec.max() + 0.5, 400)
    ax.plot(xs, gaussian_kde(rec)(xs), color="C0", lw=2, label=r"$\hat{A}_{\mathrm{LINEX}}$")
    ax.axvline(amp_db, color="r", ls="--", lw=1.2, label="ground truth")
    ax.set_xlabel("A (dB)")
    ax.set_ylabel("pdf")
    ax.set_title(f"LINEX single-slope amplitude  (N={n}, conv={conv:.0f}%)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "single_slope.png")
    fig.savefig(out, dpi=130)
    print(f"  figure saved to {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
