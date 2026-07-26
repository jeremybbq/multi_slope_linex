"""Energy-decay rates and sum-of-exponentials kernels."""

from __future__ import annotations

import numpy as np

# -ln(1e-6) = ln(1e6) ~= 13.815510557964274
_LN_1E6 = float(np.log(1e6))


def decay_rate(t_vals):
    """Return energy decay rates for decay times in seconds.

    ``t_vals`` may be scalar or array-like; ``inf`` produces a zero rate.
    """
    t = np.asarray(t_vals, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    delta = np.zeros_like(t)
    finite = np.isfinite(t)
    if np.any(t[finite] <= 0):
        raise ValueError("decay times must be positive (use inf for a flat/noise slope)")
    delta[finite] = _LN_1E6 / t[finite]
    return float(delta[0]) if scalar else delta


def decay_kernel(t_vals, time_axis, no_noise=True):
    """Build an energy-decay kernel.

    Parameters are decay times ``t_vals`` in seconds and ``(L,)`` time samples.
    Returns ``(L, d)`` columns ``exp(-decay_rate(t_vals) * time_axis)``. Set
    ``no_noise=False`` to append a constant noise column.
    """
    t_vals = np.atleast_1d(np.asarray(t_vals, dtype=float))
    time_axis = np.asarray(time_axis, dtype=float).reshape(-1, 1)  # (L, 1)
    delta = decay_rate(t_vals).reshape(1, -1)                      # (1, d)
    X = np.exp(-time_axis * delta)                                 # (L, d)
    if not no_noise:
        X = np.hstack([X, np.ones((time_axis.shape[0], 1))])
    return X
