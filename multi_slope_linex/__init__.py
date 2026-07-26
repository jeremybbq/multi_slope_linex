"""LINEX common-slope amplitude estimation."""

from __future__ import annotations

from .kernel import decay_kernel, decay_rate
from .linex import linex_loss, linex_grad_e, linex_hess_e
from .estimator import (
    constrained_linex_decay_analysis,
    common_slope_fit,
    estimate_amplitudes,
)
from .synth import generate_decaying_noise, amplitude_decay_rate

from . import pipeline
from .pipeline import common_slope_analysis

__all__ = [
    "decay_kernel",
    "decay_rate",
    "linex_loss",
    "linex_grad_e",
    "linex_hess_e",
    "constrained_linex_decay_analysis",
    "common_slope_fit",
    "estimate_amplitudes",
    "generate_decaying_noise",
    "amplitude_decay_rate",
    "pipeline",
    "common_slope_analysis",
]

__version__ = "0.1.0"
