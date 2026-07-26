"""LINEX amplitude estimation for fixed energy-decay kernels."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize, nnls

from .kernel import decay_kernel
from .linex import _g, _gp


def _obj_and_grad(w, X, logy, alpha, eps_floor):
    """Return the scaled LINEX objective and gradient."""
    Xw = X @ w
    Xw = np.maximum(Xw, eps_floor)
    e = logy - np.log(Xw)
    u = alpha * e
    scale = 2.0 / (alpha * alpha)
    obj = scale * float(np.sum(_g(u)))
    gp = _gp(u)
    grad = -(2.0 / alpha) * (X.T @ (gp / Xw))
    return obj, grad


def _warm_start(X, y):
    """Return a positive non-negative-least-squares seed."""
    d = X.shape[1]
    w0 = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w0, _ = nnls(X, y)
    except Exception:
        w0 = None
    if w0 is None or not np.all(np.isfinite(w0)) or not np.any(w0 > 0):
        with np.errstate(divide="ignore", invalid="ignore"):
            w0 = np.array([np.median(y / np.maximum(X[:, j], 1e-300)) for j in range(d)])
        w0 = np.where(np.isfinite(w0), w0, 1e-3)
    return np.maximum(w0, 1e-12)


def _alpha_schedule(alpha, n_alpha, alpha_start):
    a0 = min(alpha_start, alpha / 8.0)
    a0 = max(a0, 1e-3)
    if a0 >= alpha:
        return np.array([alpha], dtype=float)
    return np.geomspace(a0, alpha, int(n_alpha))


def _run_chain(w, X, logy, eps_floor, schedule, bounds, maxiter, ftol, gtol, info=None):
    """Solve the LINEX subproblems along an alpha schedule, warm-starting each."""
    last_success = True
    for a in schedule:
        res = minimize(
            _obj_and_grad, w, args=(X, logy, a, eps_floor), jac=True,
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol},
        )
        if np.all(np.isfinite(res.x)):
            w = np.clip(res.x, bounds[0][0], bounds[0][1])
        last_success = bool(res.success)
        if info is not None:
            info["alphas"].append(float(a))
            info["success"].append(last_success)
            info["nit"].append(int(res.nit))
            info["loss"].append(float(res.fun))
    return w, last_success


def constrained_linex_decay_analysis(
    rir2,
    kernel,
    no_noise=True,
    alpha=1.0,
    *,
    continuation=True,
    warm_start=True,
    alpha_schedule=None,
    n_alpha=7,
    alpha_start=0.05,
    ub=10.0,
    eps_rel=1e-12,
    maxiter=2000,
    ftol=1e-10,
    gtol=1e-8,
    return_info=True,
):
    """Fit non-negative amplitudes to one energy curve.

    Parameters
    ----------
    rir2 : array_like, shape (L,)
        Linear-scale instantaneous energy.
    kernel : array_like, shape (L, d)
        Fixed energy-decay basis.
    alpha : float, default 1.0
        LINEX asymmetry parameter.

    Returns
    -------
    ndarray, shape (d,), or tuple
        Energy amplitudes, with diagnostics appended when ``return_info=True``.
    """
    y = np.asarray(rir2, dtype=float).ravel()
    X = np.asarray(kernel, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"kernel rows ({X.shape[0]}) must match len(rir2) ({y.shape[0]})")
    d = X.shape[1]

    ymax = float(np.max(y)) if y.size else 0.0
    if not np.isfinite(ymax) or ymax <= 0:
        raise ValueError("rir2 must contain positive, finite energy values")
    eps_floor = eps_rel * ymax
    logy = np.log(np.maximum(y, eps_floor))
    bounds = [(0.0, float(ub))] * d

    w = _warm_start(X, np.maximum(y, eps_floor)) if warm_start else np.full(d, 1e-3)

    if continuation:
        schedule = (np.asarray(alpha_schedule, dtype=float)
                    if alpha_schedule is not None
                    else _alpha_schedule(alpha, n_alpha, alpha_start))
    else:
        schedule = np.array([alpha], dtype=float)

    info = {"alphas": [], "success": [], "nit": [], "loss": [], "fallback": False}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        w, last_success = _run_chain(
            w, X, logy, eps_floor, schedule, bounds, maxiter, ftol, gtol, info
        )

        final_obj, _ = _obj_and_grad(w, X, logy, alpha, eps_floor)

        # Multistart fallback: only if something went non-finite.
        if not (np.all(np.isfinite(w)) and np.isfinite(final_obj)):
            info["fallback"] = True
            full_sched = _alpha_schedule(alpha, max(n_alpha, 10), alpha_start)
            best_w, best_obj = None, np.inf
            for seed in (_warm_start(X, np.maximum(y, eps_floor)),
                         np.full(d, 1e-3), np.full(d, 1e-6)):
                wt, _ = _run_chain(seed, X, logy, eps_floor, full_sched,
                                   bounds, maxiter, ftol, gtol)
                jt, _ = _obj_and_grad(wt, X, logy, alpha, eps_floor)
                if np.isfinite(jt) and jt < best_obj:
                    best_obj, best_w = jt, wt
            if best_w is not None:
                w = best_w
                final_obj = best_obj
                last_success = True

    info["converged"] = bool(np.all(np.isfinite(w)) and last_success)
    info["final_loss"] = float(final_obj)
    if return_info:
        return w, info
    return w


_POOL = {}


def _pool_init(X, no_noise, alpha, kwargs):
    _POOL["X"] = X
    _POOL["no_noise"] = no_noise
    _POOL["alpha"] = alpha
    _POOL["kwargs"] = kwargs


def _pool_fit(row):
    norm = float(np.max(row))
    w, info = constrained_linex_decay_analysis(
        row / norm, _POOL["X"], no_noise=_POOL["no_noise"], alpha=_POOL["alpha"],
        **_POOL["kwargs"]
    )
    return w * norm, info


def common_slope_fit(rir2s, common_decay_times, fs, no_noise=True, alpha=1.0,
                     time_axis=None, n_jobs=1, **kwargs):
    """Fit common-slope amplitudes for multiple energy curves.

    ``rir2s`` is ``(n_curves, L)`` in linear energy units and
    ``common_decay_times`` is in seconds. Returns ``(amplitudes, infos)`` with
    amplitudes shaped ``(n_curves, d)``. Use ``n_jobs`` greater than one to fit
    curves in parallel.
    """
    rir2s = np.atleast_2d(np.asarray(rir2s, dtype=float))
    n_curves, L = rir2s.shape
    t = np.arange(L) / fs if time_axis is None else np.asarray(time_axis, dtype=float)
    X = decay_kernel(common_decay_times, t, no_noise=no_noise)
    d = X.shape[1]

    if n_jobs in (-1, 0, None):
        import os
        n_jobs = os.cpu_count() or 1
    n_jobs = min(int(n_jobs), n_curves)

    if n_jobs > 1 and n_curves > 1:
        import os
        from concurrent.futures import ProcessPoolExecutor
        _thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
        _saved_env = {v: os.environ.get(v) for v in _thread_vars}
        os.environ.update({v: "1" for v in _thread_vars})
        try:
            chunksize = max(1, n_curves // (n_jobs * 4))
            with ProcessPoolExecutor(max_workers=n_jobs, initializer=_pool_init,
                                     initargs=(X, no_noise, alpha, kwargs)) as ex:
                results = list(ex.map(_pool_fit, (rir2s[i] for i in range(n_curves)),
                                      chunksize=chunksize))
        finally:
            for v, old in _saved_env.items():
                if old is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = old
        a_vals = np.stack([r[0] for r in results])
        infos = [r[1] for r in results]
        return a_vals, infos

    a_vals = np.zeros((n_curves, d))
    infos = []
    for i in range(n_curves):
        row = rir2s[i]
        norm = float(np.max(row))
        w, info = constrained_linex_decay_analysis(
            row / norm, X, no_noise=no_noise, alpha=alpha, **kwargs
        )
        a_vals[i] = w * norm
        infos.append(info)
    return a_vals, infos


def estimate_amplitudes(signal, decay_times, fs, is_energy=False, alpha=1.0,
                        no_noise=True, time_axis=None, **kwargs):
    """Estimate amplitudes from one signal or energy curve.

    Set ``is_energy=True`` for a linear-energy input; otherwise ``signal`` is
    squared. Returns a dictionary with energy amplitudes ``w``, levels
    ``amp_db``, the decay kernel, and solver diagnostics.
    """
    x = np.asarray(signal, dtype=float).ravel()
    rir2 = x if is_energy else x ** 2
    L = rir2.shape[0]
    t = np.arange(L) / fs if time_axis is None else np.asarray(time_axis, dtype=float)
    X = decay_kernel(decay_times, t, no_noise=no_noise)
    norm = float(np.max(rir2))
    w, info = constrained_linex_decay_analysis(
        rir2 / norm, X, no_noise=no_noise, alpha=alpha, **kwargs
    )
    w = w * norm
    with np.errstate(divide="ignore"):
        amp_db = 10.0 * np.log10(np.where(w > 0, w, np.nan))
    return {"w": w, "amp_db": amp_db, "kernel": X, "info": info}
