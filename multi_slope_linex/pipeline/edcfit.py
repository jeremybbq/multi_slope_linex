"""Least-squares amplitude fitting for Schroeder energy-decay curves."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from ..kernel import decay_rate


def edc_decay_kernel(t_vals, time_axis, no_noise=True, compensated=True):
    """Build an EDC fitting kernel for fixed decay times.

    ``time_axis`` is in seconds. Set ``no_noise=False`` to include a noise
    column; set ``compensated=False`` for the uncompensated kernel.
    """
    t_vals = np.atleast_1d(np.asarray(t_vals, dtype=float))
    time_axis = np.asarray(time_axis, dtype=float).reshape(-1, 1)
    delta = decay_rate(t_vals).reshape(1, -1)
    L = time_axis.shape[0]

    if not compensated:
        exponentials = np.exp(-time_axis * delta)
        noise = np.linspace(1.0, 1.0 / L, L).reshape(-1, 1)
        return np.hstack([exponentials, noise])

    fs = L / float(time_axis[-1, 0] - time_axis[0, 0])
    expo = np.exp(-time_axis * delta)
    exponentials = (expo - expo[-1:, :]) / (1.0 - np.exp(-delta / fs))
    if no_noise:
        return exponentials
    noise = np.linspace(L - 1.0, 0.0, L).reshape(-1, 1)
    return np.hstack([exponentials, noise])


def constrained_lsq_decay_analysis(edf_norm, kernel, has_noise_col=False,
                                   fit_fraction=0.95):
    """Fit bounded amplitudes to one normalized EDC.

    ``edf_norm`` is a linear-scale ``(L,)`` curve and ``kernel`` is ``(L, d)``.
    Returns a ``(d,)`` amplitude vector.
    """
    edf = np.asarray(edf_norm, dtype=float).ravel()
    K = np.asarray(kernel, dtype=float)
    n_fit = int(np.ceil(fit_fraction * edf.shape[0]))
    Kf, ef = K[:n_fit], edf[:n_fit]
    eps = np.finfo(float).eps
    target_db = 10.0 * np.log10(np.maximum(ef, eps))

    def residual(x):
        return 10.0 * np.log10(np.maximum(Kf @ x, eps)) - target_db

    n_cols = K.shape[1]
    d = n_cols - 1 if has_noise_col else n_cols
    x0 = np.ones(n_cols)
    lower = np.zeros(n_cols)
    upper = np.full(n_cols, 10.0)
    if has_noise_col:
        x0[-1] = 1e-10
        upper[-1] = 1.0
    res = least_squares(residual, x0, bounds=(lower, upper), method="trf",
                        ftol=1e-9, xtol=1e-12, max_nfev=5000)
    return res.x


def common_slope_fit_edc(edfs, common_decay_times, fs, no_noise=True,
                         compensated=True, output_size=None):
    """Fit EDC amplitudes for a batch of fixed decay times.

    ``edfs`` has shape ``(n_curves, L)`` in linear scale. Returns
    ``(amplitudes, noise_values)`` with one amplitude row per EDC.
    """
    edfs = np.atleast_2d(np.asarray(edfs, dtype=float))
    n_curves, L = edfs.shape
    common = np.atleast_1d(np.asarray(common_decay_times, dtype=float))
    d = common.shape[0]

    if output_size is None:
        time_axis = np.linspace(0.0, (L - 1) / fs, L)
        xi = None
    else:
        time_axis = np.linspace(0.0, (L - 1) / fs, output_size)
        xi = np.linspace(0, L - 1, output_size)
    K = edc_decay_kernel(common, time_axis, no_noise=no_noise, compensated=compensated)
    has_noise = (not no_noise) or (not compensated)

    a_vals = np.zeros((n_curves, d))
    n_vals = np.zeros(n_curves)
    idx = np.arange(L)
    for i in range(n_curves):
        norm = float(edfs[i, 0])
        edf = edfs[i] / norm
        if xi is not None:
            edf = np.interp(xi, idx, edf)
        w = constrained_lsq_decay_analysis(edf, K, has_noise_col=has_noise)
        a_vals[i] = w[:d] * norm
        if has_noise:
            n_vals[i] = w[-1] * norm
    return a_vals, n_vals
