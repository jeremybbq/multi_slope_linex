"""Tests for the full RIR -> common-slope amplitude pipeline."""

import os

import numpy as np
import pytest

from multi_slope_linex import generate_decaying_noise, common_slope_analysis
from multi_slope_linex.pipeline import (
    octave_filtering,
    schroeder_int,
    rir2decay,
    determine_common_decay_times,
    hist_resolution_to_edges,
    list2map,
)
from multi_slope_linex.pipeline import decayfitnet


# --------------------------------------------------------------------------- #
# Signal front end
# --------------------------------------------------------------------------- #
def test_octave_filtering_passband():
    fs = 8000
    t = np.arange(fs) / fs
    x_in = np.sin(2 * np.pi * 1000 * t)      # in band (1000 Hz)
    x_out = np.sin(2 * np.pi * 100 * t)      # well below band
    y = octave_filtering(x_in, fs, [1000.0])
    z = octave_filtering(x_out, fs, [1000.0])
    assert y.shape == (fs, 1)
    # in-band tone retains far more energy than the out-of-band tone
    assert y[fs // 2:, 0].std() > 20 * z[fs // 2:, 0].std()


def test_schroeder_int_monotonic():
    x = np.random.default_rng(0).standard_normal(500)
    edc = schroeder_int(x)
    assert np.all(np.diff(edc) <= 1e-9)      # non-increasing
    assert np.isclose(edc[0], np.sum(x ** 2))


def test_rir2decay_squared_shape_positive():
    rng = np.random.default_rng(1)
    rir = np.exp(-3 * np.log(10) / 0.5 * np.arange(4000) / 8000) * rng.standard_normal(4000)
    decay, _ = rir2decay(rir, 8000, [1000.0], do_backwards_int=False)
    assert decay.ndim == 2 and decay.shape[1] == 1
    assert np.all(decay >= 0)


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def test_hist_edges_alignment():
    edges = hist_resolution_to_edges(0.46, 1.18, 0.05)
    assert np.isclose(edges[0], 0.45)
    assert edges[-1] >= 1.18


def test_generate_synthetic_edc_monotonic():
    from multi_slope_linex.pipeline import generate_synthetic_edc
    t = np.arange(2000) / 1000.0
    edc = generate_synthetic_edc([0.5, 1.5], [1.0, 0.3], 1e-4, t)
    assert np.all(np.diff(edc) <= 1e-9)      # Schroeder EDC is non-increasing
    assert edc[0] > edc[-1]


def test_determine_common_decay_times():
    rng = np.random.default_rng(0)
    pool = np.concatenate([
        rng.normal(0.5, 0.02, 200),
        rng.normal(1.0, 0.02, 200),
        rng.normal(2.0, 0.02, 200),
    ])
    common, clustered = determine_common_decay_times(pool, 3, hist_resolution=0.05)
    assert np.all(np.diff(common) > 0)             # sorted
    assert np.allclose(common, [0.5, 1.0, 2.0], atol=0.1)


# --------------------------------------------------------------------------- #
# Orchestrator (supplied decay times -> no heavy deps)
# --------------------------------------------------------------------------- #
def test_common_slope_analysis_given_times():
    amps, T, fs = [-10.0, -25.0], [0.4, 1.2], 8000
    h, _, _, _ = generate_decaying_noise(amps, T, fs=fs, duration_s=1.2, n=6, seed=3)
    res = common_slope_analysis(h.T, fs=fs, analysis_band=1000.0, n_common_slopes=2,
                                common_decay_times=T)
    assert res["aVals"].shape == (6, 2)
    assert res["convergence"] == 1.0
    assert np.all(res["aVals"] > 0)


def test_common_slope_analysis_multichannel():
    amps, T, fs = [-10.0, -25.0], [0.4, 1.2], 8000
    h, _, _, _ = generate_decaying_noise(amps, T, fs=fs, duration_s=1.0, n=4, seed=3)
    rirs3 = np.stack([h.T, h.T], axis=2)           # (4, L, 2)
    res = common_slope_analysis(rirs3, fs=fs, analysis_band=1000.0, n_common_slopes=2,
                                common_decay_times=T)
    assert res["aVals"].shape == (4, 2, 2)
    assert res["convergence"] == 1.0


# --------------------------------------------------------------------------- #
# EDC-fit baseline (comparison method)
# --------------------------------------------------------------------------- #
def test_edc_fit_exact_recovery_compensated():
    from multi_slope_linex.pipeline import edc_decay_kernel, common_slope_fit_edc
    fs, L = 2000, 8000
    t = np.arange(L) / fs
    T = [0.5, 1.5]
    K = edc_decay_kernel(T, t, no_noise=True, compensated=True)
    w_true = np.array([0.3, 0.01])
    edf = (K @ w_true)[None, :]
    a, n = common_slope_fit_edc(edf, T, fs)
    assert np.allclose(10 * np.log10(a[0]), 10 * np.log10(w_true), atol=0.2)
    assert np.all(n == 0)


def test_edc_fit_exact_recovery_original_conventions():
    from multi_slope_linex.pipeline import edc_decay_kernel, common_slope_fit_edc
    fs, L = 2000, 8000
    t = np.arange(L) / fs
    T = [0.5, 1.5]
    K = edc_decay_kernel(T, t, compensated=False)     # Götz original: + noise column
    w_true = np.array([0.6, 0.05, 1e-6])
    edf = (K @ w_true)[None, :]
    a, n = common_slope_fit_edc(edf, T, fs, compensated=False, output_size=100)
    assert np.allclose(10 * np.log10(a[0]), 10 * np.log10(w_true[:2]), atol=1.0)
    assert n[0] > 0


# --------------------------------------------------------------------------- #
# Maps
# --------------------------------------------------------------------------- #
def test_list2map_nearest_within_room():
    walls = [np.array([[0, 0, 0, 4], [0, 0, 4, 0], [0, 4, 4, 4], [4, 0, 4, 4]], dtype=float)]
    pos = np.array([[1.0, 1.0], [3.0, 3.0]])
    vals = np.array([10.0, 20.0])
    m, grid = list2map(vals, pos, walls, map_res=0.5)
    assert m.shape == grid["XX"].shape
    assert np.nanmin(m) >= 10.0 and np.nanmax(m) <= 20.0


# --------------------------------------------------------------------------- #
# DecayFitNet (optional: needs onnxruntime + model files)
# --------------------------------------------------------------------------- #
def test_decayfitnet_accepts_upstream_checkout_layout(tmp_path):
    checkout = tmp_path / "DecayFitNet"
    expected = checkout / "model"
    expected.mkdir(parents=True)
    assert decayfitnet._resolve_model_dir(checkout) == str(expected)


def _dfn_available():
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return bool(decayfitnet.DEFAULT_MODEL_DIR and os.path.exists(decayfitnet.DEFAULT_MODEL_DIR))


@pytest.mark.skipif(not _dfn_available(), reason="DecayFitNet model files / onnxruntime not available")
def test_decayfitnet_recovers_t60():
    rng = np.random.default_rng(0)
    fs, dur, Ttrue = 8000, 2.0, 1.0
    t = np.arange(int(dur * fs)) / fs
    rir = np.exp(-3 * np.log(10) / Ttrue * t) * rng.standard_normal(t.size)
    est = decayfitnet.DecayFitNetEstimator(fs, 1000.0, n_slopes=0)
    t_vals, a_vals, n_vals = est.estimate(rir[None, :])
    primary = t_vals[0][t_vals[0] > 0].max()
    assert abs(primary - Ttrue) < 0.2


@pytest.mark.skipif(not _dfn_available(), reason="DecayFitNet model files / onnxruntime not available")
def test_ibic_selects_order_and_recovers_times():
    from multi_slope_linex.pipeline import IbicDecayFitNet, determine_common_decay_times
    h, _, _, _ = generate_decaying_noise([-10.0, -25.0], [0.4, 1.2], fs=8000,
                                         duration_s=1.5, n=6, seed=1)
    T, A, N = IbicDecayFitNet(8000, 1000.0, max_order=3).estimate(h.T)
    assert np.all(np.sum(T > 0, axis=1) >= 2)          # 2-slope data -> >=2 slopes
    common, _ = determine_common_decay_times(T, 2, hist_resolution=0.05)
    assert np.allclose(np.sort(common), [0.4, 1.2], atol=0.3)


@pytest.mark.skipif(not _dfn_available(), reason="DecayFitNet model files / onnxruntime not available")
def test_pipeline_with_decayfitnet():
    amps, T, fs = [-10.0, -25.0], [0.4, 1.2], 8000
    h, _, _, _ = generate_decaying_noise(amps, T, fs=fs, duration_s=1.2, n=8, seed=3)
    res = common_slope_analysis(h.T, fs=fs, analysis_band=1000.0, n_common_slopes=2,
                                n_analysis_slopes=2)
    assert res["convergence"] == 1.0
    assert res["aVals"].shape == (8, 2)
    # clustered decay times should land near the true ones
    assert np.allclose(np.sort(res["commonDecayTimes"]), T, atol=0.25)
