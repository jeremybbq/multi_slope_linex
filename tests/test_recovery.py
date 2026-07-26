"""Ground-truth recovery, convergence, and the alpha-continuation fix.

These tests exercise the synthetic experiments with *known* amplitudes and assert
(a) near-unbiased recovery, (b) 100% convergence via alpha-continuation, and
(c) that a direct alpha=1 solve from the same warm start is markedly worse --
the concrete demonstration that continuation is what fixes convergence.
"""

import numpy as np
import pytest

from multi_slope_linex import (
    generate_decaying_noise,
    common_slope_fit,
    decay_kernel,
    linex_loss,
    linex_grad_e,
)
from multi_slope_linex.estimator import constrained_linex_decay_analysis as solve


def _recover(amps, Tlist, fs, dur, n, seed):
    h, _, _, _ = generate_decaying_noise(amps, Tlist, fs=fs, duration_s=dur, n=n, seed=seed)
    a_vals, infos = common_slope_fit((h ** 2).T, Tlist, fs=fs)
    rec = 10.0 * np.log10(np.maximum(a_vals, 1e-300))
    conv = np.mean([i["converged"] for i in infos])
    return rec, conv


# --------------------------------------------------------------------------- #
# Loss / kernel unit checks
# --------------------------------------------------------------------------- #
def test_linex_loss_values_and_gradient():
    assert linex_loss(0.0) == 0.0
    assert np.isclose(float(linex_loss(1.0, 1.0)), np.e - 2.0)
    # analytic gradient matches finite difference
    e, a = 0.3, 1.0
    fd = (float(linex_loss(e + 1e-6, a)) - float(linex_loss(e - 1e-6, a))) / 2e-6
    assert np.isclose(float(linex_grad_e(e, a)), fd, rtol=1e-4)
    # overflow safety: enormous residuals stay finite (the cap)
    assert np.isfinite(float(linex_loss(1e4)))
    assert np.isfinite(float(linex_grad_e(1e4)))


def test_decay_kernel_shape_and_noise_column():
    t = np.arange(100) / 1000.0
    X = decay_kernel([0.5, 1.0, np.inf], t)
    assert X.shape == (100, 3)
    assert np.allclose(X[:, -1], 1.0)   # T = inf -> constant column
    assert np.allclose(X[0], 1.0)       # t = 0 -> exp(0) = 1


# --------------------------------------------------------------------------- #
# Ground-truth recovery
# --------------------------------------------------------------------------- #
def test_single_slope_unbiased_and_converges():
    rec, conv = _recover([-10.0], [0.8], 250, 0.6, 300, seed=1)
    assert conv == 1.0
    assert abs(rec[:, 0].mean() - (-10.0)) < 0.4


def test_multi_slope_recovery_and_converges():
    rec, conv = _recover([-20.0, -20.0, -40.0], [1.0, 2.0, np.inf], 2000, 2.0, 150, seed=1)
    assert conv == 1.0
    means = rec.mean(axis=0)
    assert abs(means[0] - (-20.0)) < 1.0
    assert abs(means[1] - (-20.0)) < 1.0
    assert abs(means[2] - (-40.0)) < 0.6


def test_stress_weak_slopes_converge():
    # weak -50 dB slope + flat noise + short signal: the regime that strains fmincon
    rec, conv = _recover([-15.0, -30.0, -50.0], [0.7, 1.5, np.inf], 2000, 1.5, 100, seed=7)
    assert conv == 1.0
    assert abs(rec[:, 2].mean() - (-50.0)) < 1.0


def test_parallel_batch_matches_serial():
    # n_jobs must not change results: bit-identical amplitudes, same order.
    h, _, _, _ = generate_decaying_noise([-10.0, -25.0], [0.4, 1.2], fs=2000,
                                         duration_s=0.8, n=6, seed=0)
    rir2s = (h ** 2).T
    a_serial, infos_s = common_slope_fit(rir2s, [0.4, 1.2], 2000)
    a_par, infos_p = common_slope_fit(rir2s, [0.4, 1.2], 2000, n_jobs=2)
    assert np.array_equal(a_serial, a_par)
    assert [i["converged"] for i in infos_s] == [i["converged"] for i in infos_p]


# --------------------------------------------------------------------------- #
# The headline: alpha-continuation vs a direct alpha=1 solve (same warm start)
# --------------------------------------------------------------------------- #
def test_continuation_beats_direct_solve():
    amps, Tlist, fs, dur = [-15.0, -30.0, -50.0], [0.7, 1.5, np.inf], 2000, 1.5
    truth = np.array(amps)
    h, t, _, _ = generate_decaying_noise(amps, Tlist, fs=fs, duration_s=dur, n=100, seed=7)
    tax = np.arange(h.shape[0]) / fs

    def run(continuation):
        ok, errs = 0, []
        for i in range(h.shape[1]):
            rir2 = h[:, i] ** 2
            norm = rir2.max()
            X = decay_kernel(Tlist, tax)
            w, info = solve(rir2 / norm, X, continuation=continuation, warm_start=True)
            ok += int(info["converged"])
            errs.append(10.0 * np.log10(np.maximum(w * norm, 1e-300)) - truth)
        errs = np.array(errs)
        return ok, np.sqrt((errs ** 2).mean(axis=0)).max()

    direct_ok, direct_maxrmse = run(False)
    cont_ok, cont_maxrmse = run(True)

    # continuation: essentially perfect
    assert cont_ok >= 98
    assert cont_maxrmse < 1.5
    # direct solve from the same warm start: clearly worse (measured ~76% / ~48 dB)
    assert direct_ok <= 92
    assert direct_maxrmse > 10.0
    assert cont_ok > direct_ok


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
