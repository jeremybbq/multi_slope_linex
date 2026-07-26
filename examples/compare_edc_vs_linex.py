"""Compare LINEX and EDC amplitude maps for a supplied RIR dataset."""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multi_slope_linex import common_slope_fit, generate_decaying_noise
from multi_slope_linex.pipeline import (
    rir2decay,
    common_slope_fit_edc,
    schroeder_int,
    list2map,
)

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = REPO_DIR / "data" / "srirs.mat"
OUTPUT_DIR = REPO_DIR / "output"
DEFAULT_BANDS = [250, 500, 1000, 2000]

# Room walls of the three-coupled-rooms scenario, [x_from, y_from, x_to, y_to]
# (as in demo_calc_cs_maps_*.m)
WALLS = [
    np.array([[0, 0, 0, 8], [0, 0, 4, 0], [0, 8, 4, 8], [4, 0, 4, 2.75], [4, 4.25, 4, 8]], float),
    np.array([[4, 2, 10, 2], [10, 2, 10, 5], [4, 5, 8.5, 5]], float),
    np.array([[6, 5, 6, 13], [10, 5, 10, 13], [6, 13, 10, 13]], float),
]
# Per-slope color limits from the MATLAB demos; other bands auto-scale.
CBAR_LIMS = {
    250: [(-65, -45), (-65, -50), (-90, -70)],
    500: [(-65, -45), (-65, -50), (-90, -65)],
    1000: [(-60, -40), (-65, -50), (-80, -60)],
    2000: [(-60, -40), (-65, -50), (-80, -60)],
}


def _db(a):
    out = 10.0 * np.log10(np.maximum(a, 1e-300))
    return np.where(a > 0, out, np.nan)


def _slope_limits(band, s, db_arrays):
    preset = CBAR_LIMS.get(band)
    if preset is not None:
        return preset[s]
    vals = np.concatenate([a[:, s][np.isfinite(a[:, s])] for a in db_arrays])
    if vals.size == 0:
        return (None, None)
    lo, hi = np.percentile(vals, [2, 98])
    return (float(np.floor(lo)), float(np.ceil(hi)))


def _map_figure(band, T, rcv, a_lin, a_edc, n_receivers, out_dir):
    """2 methods x n_slopes smooth amplitude maps (matches MATLAB plotMap.m:
    pcolor + shading interp; zeros clamped to the bottom of the color scale)."""
    d = T.shape[0]
    db_arrays = (_db(a_lin), _db(a_edc))
    fig, axes = plt.subplots(2, d, figsize=(4.2 * d, 8.5), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(2, d)
    for row, (label, a_vals) in enumerate((("LINEX", a_lin), ("EDC fit", a_edc))):
        m, grid = list2map(a_vals, rcv, WALLS, map_res=0.2)
        m = m[:, :, None] if m.ndim == 2 else m
        for s in range(d):
            ax = axes[row, s]
            raw = m[:, :, s]
            md = np.where(np.isnan(raw), np.nan,
                          10.0 * np.log10(np.maximum(raw, 1e-30)))
            vmin, vmax = _slope_limits(band, s, db_arrays)
            pc = ax.pcolormesh(grid["XX"], grid["YY"], md, shading="gouraud",
                               vmin=vmin, vmax=vmax, cmap="viridis")
            for room in WALLS:
                for x0, y0, x1, y1 in room:
                    ax.plot([x0, x1], [y0, y1], "k-", lw=1.5)
            ax.set_aspect("equal")
            ax.set_title(f"{label}:  $A_{{{s + 1}}}$ (T={T[s]:.2f}s)", fontsize=10)
            fig.colorbar(pc, ax=ax, shrink=0.75, label="dB")
    fig.suptitle(f"Common-slope amplitude maps, {band} Hz — three coupled rooms "
                 f"({n_receivers} receivers)", y=0.98)
    fig.tight_layout()
    out = os.path.join(out_dir, f"compare_edc_vs_linex_{band}Hz.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def _load_omnidirectional_rirs(dataset_path, n):
    """Load channel 1 of the public MATLAB v7.3 spatial-RIR dataset."""
    import h5py

    with h5py.File(dataset_path, "r") as f:
        data = f["srirDataset"]
        srirs = data["srirs"]
        if srirs.ndim != 3:
            raise ValueError(f"expected a 3-D SRIR array, got {srirs.shape}")

        channel_axis = int(np.argmin(srirs.shape))
        time_axis = int(np.argmax(srirs.shape))
        receiver_axis = next(axis for axis in range(3)
                             if axis not in (channel_axis, time_axis))
        n_total = srirs.shape[receiver_axis]
        idx = np.unique(np.linspace(0, n_total - 1, n).astype(int))

        remaining_axes = [axis for axis in range(3) if axis != channel_axis]
        n_samples = srirs.shape[time_axis]
        rirs = np.empty((len(idx), n_samples), dtype=srirs.dtype)
        for start in range(0, n_samples, 4096):
            stop = min(start + 4096, n_samples)
            index = [slice(None)] * 3
            index[channel_axis] = 0
            index[time_axis] = slice(start, stop)
            block = np.moveaxis(
                np.asarray(srirs[tuple(index)]),
                (remaining_axes.index(receiver_axis), remaining_axes.index(time_axis)),
                (0, 1),
            )
            rirs[:, start:stop] = block[idx]

        rcv = np.asarray(data["rcvPos"])
        if rcv.shape[0] == 3:
            rcv = rcv.T
        rcv = rcv[idx, :2]
        fs = float(np.asarray(data["fs"]).squeeze()) if "fs" in data else 48000.0
    return rirs, rcv, fs


def run_real(n, bands, dataset_path, decay_times, n_jobs=1):
    rirs, rcv, fs = _load_omnidirectional_rirs(dataset_path, n)
    OUTPUT_DIR.mkdir(exist_ok=True)

    for band in bands:
        print(f"\n================ {band} Hz ================")
        T = np.asarray(decay_times, dtype=float)
        print(f"{rirs.shape[0]} receivers | T = {np.round(T, 3)} s")

        print("computing octave-band EDCs and squared RIRs ...")
        edfs, frir2s = [], []
        for r in rirs:
            ds, _ = rir2decay(r, fs, [band], do_backwards_int=False)
            frir2s.append(ds[:, 0])
            edfs.append(np.cumsum(ds[::-1, 0])[::-1])
        L = min(len(e) for e in edfs)
        edfs = np.stack([e[:L] for e in edfs])
        frir2s = np.stack([e[:L] for e in frir2s])

        t0 = time.time()
        a_edc, _ = common_slope_fit_edc(edfs, T, fs, n_jobs=n_jobs)
        print(f"EDC fit done ({time.time() - t0:.0f}s)")
        t0 = time.time()
        a_lin, infos = common_slope_fit(frir2s, T, fs, n_jobs=n_jobs)
        conv = 100 * np.mean([i["converged"] for i in infos])
        print(f"LINEX fit done ({time.time() - t0:.0f}s), convergence {conv:.0f}%")

        db_edc, db_lin = _db(a_edc), _db(a_lin)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)   # all-NaN slope columns
            print("per-slope medians (dB):")
            print("  EDC   :", np.round(np.nanmedian(db_edc, 0), 1))
            print("  LINEX :", np.round(np.nanmedian(db_lin, 0), 1))
            print("  gap (EDC − LINEX):", np.round(np.nanmedian(db_edc - db_lin, 0), 2))
        out = _map_figure(band, T, rcv, a_lin, a_edc, rirs.shape[0], OUTPUT_DIR)
        print(f"figure saved to {out}")


def run_synthetic(n=200):
    print("dataset not found — running the synthetic fallback comparison")
    amps, T, fs, dur = [-10.0, -25.0], [0.5, 1.5], 4000, 2.0
    h, _, _, _ = generate_decaying_noise(amps, T, fs=fs, duration_s=dur, n=n, seed=1)
    frir2s = (h ** 2).T
    edfs = np.stack([schroeder_int(h[:, i]) for i in range(n)])
    a_edc, _ = common_slope_fit_edc(edfs, T, fs)
    a_lin, infos = common_slope_fit(frir2s, T, fs)
    conv = 100 * np.mean([i["converged"] for i in infos])
    db_edc, db_lin = _db(a_edc), _db(a_lin)
    print(f"truth {amps} dB | LINEX convergence {conv:.0f}%")
    print("  EDC   mean:", np.round(np.nanmean(db_edc, 0), 2), " std:", np.round(np.nanstd(db_edc, 0), 2))
    print("  LINEX mean:", np.round(np.nanmean(db_lin, 0), 2), " std:", np.round(np.nanstd(db_lin, 0), 2))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for s in range(2):
        axes[s].hist(db_edc[:, s], bins=30, alpha=0.5, density=True, label="EDC fit")
        axes[s].hist(db_lin[:, s], bins=30, alpha=0.5, density=True, label="LINEX")
        axes[s].axvline(amps[s], color="r", ls="--", lw=1.2, label="truth")
        axes[s].set_xlabel(f"$A_{{{s + 1}}}$ (dB), T={T[s]}s")
        axes[s].legend(fontsize=8)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / "compare_edc_vs_linex_synthetic.png"
    fig.savefig(out, dpi=130)
    print(f"figure saved to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=838, help="number of receivers (real data)")
    ap.add_argument("--bands", type=int, nargs="+", default=DEFAULT_BANDS,
                    help="octave band centers in Hz")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH,
                    help="path to the public srirs.mat dataset")
    ap.add_argument("--decay-times", type=float, nargs="+", default=[0.73, 1.43, 3.48],
                    help="common decay times in seconds for the RIR dataset")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes for the LINEX fit (-1 = all cores)")
    ap.add_argument("--synthetic", action="store_true", help="run the synthetic fallback")
    args = ap.parse_args()
    if args.synthetic:
        run_synthetic()
    elif not args.dataset.exists():
        raise FileNotFoundError(f"RIR dataset not found: {args.dataset}")
    else:
        run_real(args.n, args.bands, args.dataset, args.decay_times, n_jobs=args.jobs)
