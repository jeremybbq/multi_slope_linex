"""Numerically stable LINEX loss and derivatives."""

from __future__ import annotations

import numpy as np

_U_CAP = 30.0


def _g(u):
    """Evaluate the capped LINEX helper ``exp(u) - u - 1``."""
    u = np.asarray(u, dtype=float)
    uc = np.minimum(u, _U_CAP)
    base = np.expm1(uc) - uc                       # exact for u <= cap
    gp_cap = np.expm1(_U_CAP)                       # slope at the cap
    extra = np.where(u > _U_CAP, gp_cap * (u - _U_CAP), 0.0)
    return base + extra


def _gp(u):
    """Evaluate the first derivative of the capped LINEX helper."""
    u = np.asarray(u, dtype=float)
    uc = np.minimum(u, _U_CAP)
    return np.expm1(uc)


def _gpp(u):
    """Evaluate the second derivative of the capped LINEX helper."""
    u = np.asarray(u, dtype=float)
    uc = np.minimum(u, _U_CAP)
    return np.exp(uc)


def linex_loss(e, alpha=1.0):
    """Return elementwise LINEX loss values."""
    return _g(alpha * np.asarray(e, dtype=float))


def linex_grad_e(e, alpha=1.0):
    """Return the LINEX derivative with respect to residuals."""
    return alpha * _gp(alpha * np.asarray(e, dtype=float))


def linex_hess_e(e, alpha=1.0):
    """Return the LINEX second derivative with respect to residuals."""
    return (alpha ** 2) * _gpp(alpha * np.asarray(e, dtype=float))
