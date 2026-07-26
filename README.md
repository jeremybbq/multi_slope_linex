# multi_slope_linex

**Multi-slope amplitude estimation** for room-acoustic energy decay. Fit the non-negative amplitudes of a multi-exponential decay model to a room impulse response with a **LINEX (linear-exponential) loss**.

For the model, statistical derivation, and evaluation, see Bai and Schlecht, [*Estimation of Multi-Slope Amplitudes in Late Reverberation*](https://www.dafx.de/paper-archive/2025/DAFx25_paper_28.pdf), Proceedings of the 28th International Conference on Digital Audio Effects (DAFx25), 2025, pp. 194–201.

A concise summary of the math details are also available in the notes belows:

- [Inferences of amplitude estimator](docs/AMPLITUDE.md)
- [Convergence with robust α-continuation](docs/CONTINUATION.md)

## Quick Start

### Install

From the repository root:

```bash
pip install .
pip install ".[examples]"   # example plots
pip install ".[pipeline]"   # automatic decay-time estimation
```

### Estimate Amplitudes

Use `estimate_amplitudes` when the decay times are known.

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

### Common Slope Analysis of RIRs Batch

Use `common_slope_analysis` to analyze a set of room impulse responses.

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
