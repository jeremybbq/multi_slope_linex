"""Pipeline for common-slope analysis of room impulse responses."""

from __future__ import annotations

from .signal import (
    octave_filtering,
    schroeder_int,
    rir_onset,
    rir2decay,
    rir2decay_batch,
    generate_synthetic_edc,
    pow2db,
    db2pow,
)
from .cluster import determine_common_decay_times, hist_resolution_to_edges
from .analysis import common_slope_analysis
from .maps import list2map, is_inside_boundary, is_on_boundary
from .decayfitnet import DecayFitNetEstimator, IbicDecayFitNet, preprocess_rir_to_edc
from .edcfit import edc_decay_kernel, constrained_lsq_decay_analysis, common_slope_fit_edc

__all__ = [
    "octave_filtering",
    "schroeder_int",
    "rir_onset",
    "rir2decay",
    "rir2decay_batch",
    "generate_synthetic_edc",
    "pow2db",
    "db2pow",
    "determine_common_decay_times",
    "hist_resolution_to_edges",
    "common_slope_analysis",
    "list2map",
    "is_inside_boundary",
    "is_on_boundary",
    "DecayFitNetEstimator",
    "IbicDecayFitNet",
    "preprocess_rir_to_edc",
    "edc_decay_kernel",
    "constrained_lsq_decay_analysis",
    "common_slope_fit_edc",
]
