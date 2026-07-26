"""Full pipeline demo: RIRs -> octave filter -> decay times -> LINEX amplitudes.

By default it runs on synthetic broadband decaying-noise RIRs with known slopes.
If DecayFitNet is available (onnxruntime + model files, via DFN_MODEL_DIR
or --model-dir) it estimates the common decay times automatically; otherwise it
falls back to the known decay times.

Run:
    python examples/pipeline_demo.py
    DFN_MODEL_DIR=/path/to/DecayFitNet python examples/pipeline_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multi_slope_linex import generate_decaying_noise, common_slope_analysis
from multi_slope_linex.pipeline import rir2decay, decayfitnet
from multi_slope_linex import decay_kernel


def main():
    amps = [-10.0, -25.0]          # dB
    T = [0.4, 1.2]                 # s
    fs, dur, band = 8000, 1.2, 1000.0
    h, _, _, _ = generate_decaying_noise(amps, T, fs=fs, duration_s=dur, n=8, seed=3)
    rirs = h.T                     # (8, L)

    # Use DecayFitNet if its model files are reachable; else supply the known times.
    use_dfn = bool(decayfitnet.DEFAULT_MODEL_DIR and os.path.exists(decayfitnet.DEFAULT_MODEL_DIR))
    kwargs = {"n_analysis_slopes": 2} if use_dfn else {"common_decay_times": T}
    print(f"Decay times from: {'DecayFitNet' if use_dfn else 'supplied ground truth'}")

    res = common_slope_analysis(rirs, fs=fs, analysis_band=band, n_common_slopes=2, **kwargs)

    print(f"common decay times : {np.round(res['commonDecayTimes'], 3)} s  (truth {T})")
    print(f"convergence        : {100 * res['convergence']:.0f}%")
    amp_db = 10 * np.log10(np.maximum(res["aVals"], 1e-300))
    # NB: amplitudes are in the analysis band's filtered-energy units, so they sit
    # below the broadband input level by the octave filter's energy fraction
    # (~-10 dB here). The slope-to-slope *difference* is preserved:
    print(f"mean amplitudes    : {np.round(amp_db.mean(0), 2)} dB (in-band level)")
    print(f"slope difference   : {amp_db.mean(0)[0] - amp_db.mean(0)[1]:+.2f} dB  "
          f"(truth {amps[0] - amps[1]:+.0f} dB)")

    # Example fit: squared filtered RIR vs the fitted common-slope model.
    frir2, _ = rir2decay(rirs[0], fs, [band], do_backwards_int=False, analyse_full_rir=True)
    frir2 = frir2[:, 0]
    t = np.arange(frir2.shape[0]) / fs
    X = decay_kernel(res["commonDecayTimes"], t)
    model = X @ res["aVals"][0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, 10 * np.log10(np.maximum(frir2, 1e-300)), color="0.8", lw=0.6,
            label="squared filtered RIR")
    ax.plot(t, 10 * np.log10(np.maximum(model, 1e-300)), color="C3", lw=2,
            label="fitted common-slope model")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("energy (dB)")
    ax.set_title(f"Pipeline fit @ {band:.0f} Hz  (convergence {100 * res['convergence']:.0f}%)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_demo.png")
    fig.savefig(out, dpi=130)
    print(f"figure saved to {out}")


if __name__ == "__main__":
    main()
