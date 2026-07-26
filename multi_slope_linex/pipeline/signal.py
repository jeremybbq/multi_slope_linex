"""Signal preprocessing for RIR energy-decay analysis."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, spectrogram

_SQRT_1P5 = float(np.sqrt(1.5))


def pow2db(x):
    """Convert linear power to decibels."""
    return 10.0 * np.log10(x)


def db2pow(x):
    """Convert decibels to linear power."""
    return 10.0 ** (np.asarray(x, dtype=float) / 10.0)


def octave_filtering(input_signal, fs, f_bands, bandwidth_factor=_SQRT_1P5, order=5):
    """Filter a signal into octave bands.

    ``f_bands`` contains center frequencies in Hz. Returns filtered samples with
    shape ``(L, n_bands)``; zero and Nyquist centers select edge bands.
    """
    x = np.asarray(input_signal, dtype=float).ravel()
    f_bands = np.atleast_1d(np.asarray(f_bands, dtype=float))
    out = np.zeros((x.shape[0], f_bands.shape[0]))
    for b, fc in enumerate(f_bands):
        if fc == 0:
            f_cut = (1.0 / bandwidth_factor) * f_bands[b + 1]
            sos = butter(order, f_cut, btype="lowpass", fs=fs, output="sos")
        elif fc == fs / 2:
            prev = f_bands[b - 1] if b > 0 else fc / 2.0
            sos = butter(order, bandwidth_factor * prev, btype="highpass", fs=fs, output="sos")
        else:
            edges = fc * np.array([1.0 / bandwidth_factor, bandwidth_factor])
            sos = butter(order, edges, btype="bandpass", fs=fs, output="sos")
        out[:, b] = sosfilt(sos, x)
    return out


def schroeder_int(rir, upper_lim=None):
    """Return backward-integrated energy for a signal or sample matrix."""
    x = np.asarray(rir, dtype=float)
    if upper_lim is None:
        upper_lim = x.shape[0]
    flipped = x[:upper_lim][::-1]
    integrated = np.cumsum(flipped ** 2, axis=0)
    return integrated[::-1]


def rir_onset(rir, fs=1.0):
    """Return the estimated onset index of a one-dimensional RIR."""
    x = np.asarray(rir, dtype=float).ravel()
    _, _, S = spectrogram(x, fs=fs, window="hamming", nperseg=64, noverlap=60,
                          nfft=64, mode="magnitude")
    windowed_energy = np.sum(np.abs(S), axis=0)
    delta = windowed_energy[1:] / (windowed_energy[:-1] + 1e-16)
    onset_frame = int(np.argmax(delta))
    return (onset_frame - 2) * 4 + 64


def generate_synthetic_edc(t_vals, a_vals, n_val, time_axis, compensate_uli=True):
    """Generate a Schroeder energy-decay curve.

    ``t_vals`` and ``a_vals`` give decay times and amplitudes per slope;
    ``n_val`` is the noise level. Returns a curve shaped like ``time_axis``.
    """
    t = np.atleast_1d(np.asarray(t_vals, dtype=float))
    a = np.atleast_1d(np.asarray(a_vals, dtype=float))
    time_axis = np.asarray(time_axis, dtype=float)
    L = time_axis.shape[0]

    tau = np.zeros_like(t)
    active = t != 0
    tau[active] = np.log(1e6) / t[active]
    exps = np.exp(-time_axis[:, None] * tau[None, :])           # (L, k)
    if compensate_uli:
        exps = exps - exps[-1:, :]
    exps = exps * a[None, :]
    exps[:, ~active] = 0.0
    noise = float(n_val) * np.arange(L, 0, -1)
    return exps.sum(axis=1) + noise


def rir2decay(rir, fs, f_bands, do_backwards_int=False, analyse_full_rir=True,
              normalize=False, bandwidth_factor=_SQRT_1P5):
    """Convert one RIR into per-band energy or EDC curves.

    Returns ``(decay, norm_values)``. ``decay`` has shape ``(L, n_bands)``;
    ``do_backwards_int=True`` selects Schroeder integration and ``normalize=True``
    returns per-band normalization values.
    """
    rir = np.asarray(rir, dtype=float).ravel()

    nz = np.nonzero(rir)[0]
    if nz.size:
        rir = rir[: nz[-1] + 1]

    rir_bands = octave_filtering(rir, fs, f_bands, bandwidth_factor=bandwidth_factor)

    n_discard = int(np.ceil(5e-3 * rir.shape[0]))
    if n_discard > 0:
        rir_bands = rir_bands[:-n_discard, :]

    t0 = 0 if analyse_full_rir else max(rir_onset(rir, fs), 0)
    rir_bands = rir_bands[t0:, :]

    if do_backwards_int:
        decay = schroeder_int(rir_bands)
    else:
        decay = rir_bands ** 2

    norm_vals = None
    if normalize:
        norm_vals = np.max(np.abs(decay), axis=0)
        decay = decay / norm_vals
    return decay, norm_vals


def rir2decay_batch(rirs, fs, f_bands, do_backwards_int=False, analyse_full_rir=True,
                    normalize=False, bandwidth_factor=_SQRT_1P5):
    """Apply ``rir2decay`` to each row of an ``(n_rirs, L)`` batch.

    Returns lists of decay arrays and normalization values.
    """
    rirs = np.atleast_2d(np.asarray(rirs, dtype=float))
    all_decays, all_norms = [], []
    for i in range(rirs.shape[0]):
        decay, nv = rir2decay(rirs[i], fs, f_bands, do_backwards_int,
                              analyse_full_rir, normalize, bandwidth_factor)
        all_decays.append(decay)
        all_norms.append(nv)
    return all_decays, all_norms
