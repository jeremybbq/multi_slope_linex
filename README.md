# multi_slope_linex

**Multi-slope amplitude estimation** for room-acoustic energy decay. Fit the non-negative amplitudes of a multi-exponential decay model to a room impulse response with a **LINEX (linear-exponential) loss**.

Use `estimate_amplitudes` when the decay times are known, or `common_slope_analysis` to analyze a set of room impulse responses.

## References

For the model, statistical derivation, and evaluation, see Bai and Schlecht, [*Estimation of Multi-Slope Amplitudes in Late Reverberation*](https://www.dafx.de/paper-archive/2025/DAFx25_paper_28.pdf), Proceedings of the 28th International Conference on Digital Audio Effects (DAFx25), 2025, pp. 194–201.

Repository-specific details are split into two concise notes:

- [Basic estimator mathematics](docs/LINEX_AMPLITUDE_MATH.md)
- [Robust α-continuation](docs/LINEX_CONVERGENCE_FIX.md)

## Install

From the repository root:

```bash
python -m pip install .
python -m pip install ".[examples]"   # example plots
python -m pip install ".[pipeline]"   # automatic decay-time estimation
```

## Quick Start

### Estimate Amplitudes

```python
import numpy as np
from multi_slope_linex import generate_decaying_noise, estimate_amplitudes

# A 3-slope decaying-noise impulse response:
# amplitudes = [-20, -20, -40] dB; decay times = [1, 2, inf] s
h, _, _, _ = generate_decaying_noise(
    [-20, -20, -40], [1, 2, np.inf], fs=2000, duration_s=2
)

result = estimate_amplitudes(
    h[:, 0], decay_times=[1, 2, np.inf], fs=2000
)
print(result["amp_db"])
```

For several energy curves:

```python
from multi_slope_linex import common_slope_fit

amplitudes, info = common_slope_fit(
    (h**2).T, common_decay_times=[1, 2, np.inf], fs=2000
)
```

`common_slope_fit` accepts energy (`h²`). `estimate_amplitudes` accepts a raw signal and squares it by default.

### Common Slope Analysis of a Set of RIRs

```python
from multi_slope_linex import common_slope_analysis

# rirs: (n_rirs, samples) or (n_rirs, samples, channels)
result = common_slope_analysis(
    rirs,
    fs=32000,
    analysis_band=1000.0,
    n_common_slopes=3,
)

print(result["commonDecayTimes"])  # shared decay times in seconds
print(result["aVals"])             # energy amplitudes, one row per RIR
```

For automatic decay-time estimation, clone [DecayFitNet](https://github.com/georg-goetz/DecayFitNet) and set `DFN_MODEL_DIR` to either the checkout or its `model/` directory. When the decay times are already known, pass them with `common_decay_times=` and no model files are required.

## Examples

```bash
python examples/single_slope.py [N]     # one decay component
python examples/multi_slope.py [N]      # three decay components
python examples/pipeline_demo.py        # a synthetic RIR set
python examples/compare_edc_vs_linex.py # LINEX and EDC amplitude maps
```

The example scripts save their figures alongside the script files.
