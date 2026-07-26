"""Common-slope analysis for room impulse responses."""

from __future__ import annotations

import numpy as np

from ..estimator import common_slope_fit
from .signal import rir2decay_batch, _SQRT_1P5
from .cluster import determine_common_decay_times


def _reshape_to_batch(rirs):
    """(nRIRs, L) or (nRIRs, L, nChannels) -> ((nRIRs*nChannels, L), nRIRs, nChannels)."""
    rirs = np.asarray(rirs, dtype=float)
    if rirs.ndim == 2:
        return rirs, rirs.shape[0], 1
    if rirs.ndim == 3:
        n_rirs, L, n_ch = rirs.shape
        batch = np.transpose(rirs, (0, 2, 1)).reshape(n_rirs * n_ch, L)
        return batch, n_rirs, n_ch
    raise ValueError("rirs must be (nRIRs, L) or (nRIRs, L, nChannels)")


def common_slope_analysis(
    rirs,
    fs,
    analysis_band,
    n_common_slopes=3,
    *,
    common_decay_times=None,
    decay_time_estimator=None,
    n_analysis_slopes=3,
    hist_resolution=0.05,
    model_dir=None,
    dfn_mode="ibic",
    dfn_n_slopes=None,
    alpha=1.0,
    no_noise=True,
    bandwidth_factor=_SQRT_1P5,
    **solver_kwargs,
):
    """Analyze a set of RIRs with shared decay times.

    ``rirs`` is ``(n_rirs, L)`` or ``(n_rirs, L, n_channels)``. Supply
    ``common_decay_times`` directly, a ``decay_time_estimator``, or configure
    DecayFitNet. Returns a dictionary with ``aVals``, ``commonDecayTimes``,
    ``tVals_standard``, ``convergence``, ``infos``, and ``analysisBand``.
    """
    batch, n_rirs, n_ch = _reshape_to_batch(rirs)

    decays, _ = rir2decay_batch(batch, fs, [analysis_band], do_backwards_int=False,
                                analyse_full_rir=True, normalize=False,
                                bandwidth_factor=bandwidth_factor)
    min_len = min(d.shape[0] for d in decays)
    frir2s = np.stack([d[:min_len, 0] for d in decays], axis=0)  # (N, min_len)

    t_standard = None
    if common_decay_times is not None:
        common = np.atleast_1d(np.asarray(common_decay_times, dtype=float))
    else:
        if decay_time_estimator is not None:
            t_standard = np.asarray(decay_time_estimator(batch), dtype=float)
        else:
            try:
                if dfn_mode == "ibic":
                    from .decayfitnet import IbicDecayFitNet
                    est = IbicDecayFitNet(fs, analysis_band, max_order=n_analysis_slopes,
                                          model_dir=model_dir)
                elif dfn_mode in ("fixed", "auto"):
                    from .decayfitnet import DecayFitNetEstimator
                    ns = (0 if dfn_mode == "auto"
                          else (n_analysis_slopes if dfn_n_slopes is None else dfn_n_slopes))
                    est = DecayFitNetEstimator(fs, analysis_band, n_slopes=ns,
                                               model_dir=model_dir)
                else:
                    raise ValueError(f"unknown dfn_mode {dfn_mode!r}")
                t_standard, _, _ = est.estimate(batch)
            except Exception as exc:
                raise RuntimeError(
                    "Could not estimate decay times automatically: "
                    f"{exc}. Pass `common_decay_times=...` explicitly, or a "
                    "`decay_time_estimator` callable, or install onnxruntime and "
                    "point `model_dir`/DFN_MODEL_DIR at the DecayFitNet models."
                ) from exc
        common, _ = determine_common_decay_times(t_standard, n_common_slopes,
                                                  hist_resolution=hist_resolution)

    a_vals, infos = common_slope_fit(frir2s, common, fs, no_noise=no_noise,
                                     alpha=alpha, **solver_kwargs)

    d = a_vals.shape[1]
    a_out = a_vals if n_ch == 1 else a_vals.reshape(n_rirs, n_ch, d)
    if t_standard is not None and n_ch > 1:
        t_standard = t_standard.reshape(n_rirs, n_ch, -1)

    return {
        "aVals": a_out,
        "commonDecayTimes": common,
        "tVals_standard": t_standard,
        "convergence": float(np.mean([i["converged"] for i in infos])),
        "infos": infos,
        "analysisBand": analysis_band,
    }
