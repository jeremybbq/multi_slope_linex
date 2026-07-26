"""DecayFitNet-based per-RIR decay-time estimation."""

from __future__ import annotations

import os
import pickle

import numpy as np
from scipy.signal import butter, sosfilt
from scipy.special import gammaln

DEFAULT_MODEL_DIR = os.environ.get("DFN_MODEL_DIR")
_SQRT2 = float(np.sqrt(2.0))


def _resolve_model_dir(model_dir):
    """Return the model directory from a model directory or upstream checkout."""
    model_dir = os.fspath(model_dir)
    checkout_model_dir = os.path.join(model_dir, "model")
    if os.path.isdir(checkout_model_dir):
        return checkout_model_dir
    return model_dir


def _discard_last_n_percent(x, n_percent):
    last = int(round((1.0 - n_percent / 100.0) * x.shape[0]))
    return x[:last]


def _octave_sos(band, fs, order=5):
    if float(band) * _SQRT2 >= fs / 2:
        return butter(order, float(band) / _SQRT2, btype="highpass", fs=fs, output="sos")
    edges = float(band) * np.array([1.0 / _SQRT2, _SQRT2])
    return butter(order, edges, btype="bandpass", fs=fs, output="sos")


def preprocess_rir_to_edc(rir, fs, band, normfactor, output_size=100):
    """Convert one RIR to a DecayFitNet EDC input.

    Returns ``(edc, time_scale, noise_scale, normalization)`` where ``edc`` has
    ``output_size`` samples.
    """
    rir = np.asarray(rir, dtype=float).ravel()
    nz = np.nonzero(rir)[0]
    if nz.size:
        rir = rir[: nz[-1] + 1]

    x = sosfilt(_octave_sos(band, fs), rir)
    x = _discard_last_n_percent(x, 0.5)

    edc = np.cumsum((x[::-1]) ** 2)[::-1]
    norm_val = float(edc.max())
    edc = edc / norm_val
    edc_db = 10.0 * np.log10(edc + 1e-10)

    l1 = edc_db.shape[0]
    n_adjust = l1 / output_size
    t_adjust = 10.0 / (l1 / fs)

    edc_db = _discard_last_n_percent(edc_db, 5)
    xi = np.linspace(0, edc_db.shape[0] - 1, output_size)
    edc_db = np.interp(xi, np.arange(edc_db.shape[0]), edc_db)
    edc_db = 2.0 * edc_db / normfactor + 1.0
    return edc_db.astype(np.float32), t_adjust, n_adjust, norm_val


class DecayFitNetEstimator:
    """Estimate decay parameters with a DecayFitNet ONNX model.

    Provide the RIR sampling rate, octave-band center frequency, and a directory
    containing the ONNX model and its input transform. ``n_slopes=0`` selects
    the combined model; positive values select a fixed-order model.
    """

    def __init__(self, fs, filter_frequency, n_slopes=0, model_dir=None, output_size=100):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "DecayFitNetEstimator needs onnxruntime. Install it with "
                "`pip install onnxruntime`."
            ) from exc

        self.fs = fs
        self.filter_frequency = filter_frequency
        self.n_slopes = n_slopes
        self.output_size = output_size
        model_dir = model_dir or DEFAULT_MODEL_DIR
        if model_dir is None:
            raise FileNotFoundError(
                "Set DFN_MODEL_DIR or pass model_dir with the DecayFitNet models."
            )
        model_dir = _resolve_model_dir(model_dir)
        suffix = "" if n_slopes == 0 else f"_{n_slopes}slopes"
        onnx_path = os.path.join(model_dir, f"DecayFitNet{suffix}_v10.onnx")
        pkl_path = os.path.join(model_dir, f"input_transform{suffix}.pkl")
        if not (os.path.exists(onnx_path) and os.path.exists(pkl_path)):
            raise FileNotFoundError(
                f"DecayFitNet model files not found in {model_dir!r} "
                f"(need {os.path.basename(onnx_path)} and {os.path.basename(pkl_path)}). "
                "Set DFN_MODEL_DIR or pass model_dir."
            )
        self._session = ort.InferenceSession(onnx_path)
        self._input_name = self._session.get_inputs()[0].name
        self._normfactor = float(np.asarray(pickle.load(open(pkl_path, "rb"))["edcs_db_normfactor"]))

    def estimate(self, rirs):
        """Estimate parameters for ``(n_rirs, L)`` RIRs.

        Returns ``(decay_times, amplitudes, noise_values)``. Decay times and
        amplitudes have one row per RIR; noise values are one-dimensional.
        """
        rirs = np.atleast_2d(np.asarray(rirs, dtype=float))
        edcs, t_adj, n_adj = [], [], []
        for i in range(rirs.shape[0]):
            edc, ta, na, _ = preprocess_rir_to_edc(
                rirs[i], self.fs, self.filter_frequency, self._normfactor, self.output_size)
            edcs.append(edc)
            t_adj.append(ta)
            n_adj.append(na)
        edcs = np.stack(edcs, axis=0)
        t_adj = np.asarray(t_adj)[:, None]
        n_adj = np.asarray(n_adj)[:, None]

        outs = self._session.run(None, {self._input_name: edcs})
        t_pred, a_pred, n_exp = outs[0].copy(), outs[1].copy(), outs[2].copy()

        exactly_n = self.n_slopes != 0
        if not exactly_n:
            probs = outs[3]
            n_count = np.argmax(probs, axis=1) + 1
            idx = np.tile(np.arange(1, 4), (a_pred.shape[0], 1))
            a_pred[n_count[:, None] < idx] = 0.0

        n_vals = np.power(10.0, np.clip(n_exp, -32, 32)).ravel() / n_adj.ravel()
        t_pred = t_pred / t_adj

        if not exactly_n:
            mask = a_pred == 0
            t_pred[mask] = np.nan
            a_pred[mask] = np.nan
        order = np.argsort(t_pred, axis=1)
        t_pred = np.take_along_axis(t_pred, order, axis=1)
        a_pred = np.take_along_axis(a_pred, order, axis=1)
        if not exactly_n:
            t_pred = np.nan_to_num(t_pred, nan=0.0)
            a_pred = np.nan_to_num(a_pred, nan=0.0)
        return t_pred, a_pred, n_vals


class IbicDecayFitNet:
    """Select DecayFitNet slope counts per RIR with IBIC.

    The constructor creates models for orders one through ``max_order``.
    """

    def __init__(self, fs, filter_frequency, max_order=3, model_dir=None, output_size=100):
        self.fs = fs
        self.filter_frequency = filter_frequency
        self.max_order = max_order
        self.output_size = output_size
        self.nets = {
            o: DecayFitNetEstimator(fs, filter_frequency, n_slopes=o,
                                    model_dir=model_dir, output_size=output_size)
            for o in range(1, max_order + 1)
        }

    def _resample100(self, y):
        L = y.shape[0]
        return np.interp(np.linspace(0, L - 1, self.output_size), np.arange(L), y)

    def estimate(self, rirs):
        """Return IBIC-selected parameters for an ``(n_rirs, L)`` batch.

        Decay times and amplitudes are zero-padded to ``max_order`` columns.
        """
        from .signal import rir2decay, generate_synthetic_edc, pow2db

        rirs = np.atleast_2d(np.asarray(rirs, dtype=float))
        n = rirs.shape[0]
        n95 = int(round(0.95 * self.output_size))

        per_order = {o: self.nets[o].estimate(rirs) for o in range(1, self.max_order + 1)}

        t_out = np.zeros((n, self.max_order))
        a_out = np.zeros((n, self.max_order))
        n_out = np.zeros(n)
        for i in range(n):
            edc, _ = rir2decay(rirs[i], self.fs, [self.filter_frequency],
                               do_backwards_int=True, analyse_full_rir=True, normalize=True)
            edc = edc[:, 0]
            L = edc.shape[0]
            time_axis = np.arange(L) / self.fs
            true_ds = self._resample100(pow2db(np.maximum(edc, 1e-300)))

            best_ibic, best_o = -np.inf, 1
            for o in range(1, self.max_order + 1):
                t_o, a_o, n_o = per_order[o]
                fitted = generate_synthetic_edc(t_o[i], a_o[i], n_o[i], time_axis,
                                                compensate_uli=True)
                fit_ds = self._resample100(pow2db(np.maximum(fitted, 1e-300)))
                sse = float(np.sum((fit_ds[:n95] - true_ds[:n95]) ** 2))
                log_lik = np.log(0.5) + gammaln(n95 / 2) - (n95 / 2) * np.log(np.pi * sse + 1e-300)
                ibic = 2 * log_lik - (2 * o + 1) * np.log(n95)
                if ibic > best_ibic:
                    best_ibic, best_o = ibic, o
            t_out[i, :best_o] = per_order[best_o][0][i]
            a_out[i, :best_o] = per_order[best_o][1][i]
            n_out[i] = per_order[best_o][2][i]
        return t_out, a_out, n_out
