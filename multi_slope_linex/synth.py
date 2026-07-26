"""Synthetic multi-slope decaying-noise signals."""

from __future__ import annotations

import numpy as np


def amplitude_decay_rate(decay_times_s):
    """Amplitude decay rate ``ln(1000) / T`` (``inf`` -> 0)."""
    T = np.atleast_1d(np.asarray(decay_times_s, dtype=float))
    rate = np.zeros_like(T)
    finite = np.isfinite(T)
    rate[finite] = np.log(1000.0) / T[finite]
    return rate


def generate_decaying_noise(amps_db, decay_times_s, fs, duration_s, n=1, seed=None):
    """Generate multi-slope decaying-noise signals.

    ``amps_db`` and ``decay_times_s`` define one value per slope. Returns
    ``(signals, time_axis, energy_amplitudes, decay_times)`` with signals shaped
    ``(L, n)`` and energy amplitudes in linear units.
    """
    rng = np.random.default_rng(seed)
    amps_db = np.atleast_1d(np.asarray(amps_db, dtype=float))
    T = np.atleast_1d(np.asarray(decay_times_s, dtype=float))
    if amps_db.shape != T.shape:
        raise ValueError("amps_db and decay_times_s must have the same length")

    L = int(round(duration_s * fs))
    t = np.arange(L) / fs
    A = 10.0 ** (amps_db / 20.0)
    decay = amplitude_decay_rate(T)

    h = np.zeros((L, n))
    for j in range(A.shape[0]):
        env = np.exp(-decay[j] * t)[:, None]          # (L, 1)
        z = rng.standard_normal((L, n))
        h += env * z * A[j]

    return h, t, A ** 2, T
