"""Clustering utilities for common decay times."""

from __future__ import annotations

import numpy as np


def hist_resolution_to_edges(min_val, max_val, hist_resolution):
    """Return histogram edges covering a range at the requested resolution."""
    hist_min = np.floor(min_val / hist_resolution) * hist_resolution
    hist_max = np.ceil(max_val / hist_resolution) * hist_resolution
    return np.arange(hist_min, hist_max + hist_resolution / 2, hist_resolution)


def _kmedians_1d(x, k, n_init=5, max_iter=10000, seed=0):
    """1-D k-medians (city-block k-means). Returns (labels, centers)."""
    x = np.asarray(x, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    best_labels, best_centers, best_cost = None, None, np.inf
    uniq = np.unique(x)
    if uniq.size <= k:
        centers = np.sort(np.concatenate([uniq, np.full(k - uniq.size, uniq[-1])]))
        labels = np.array([int(np.argmin(np.abs(centers - v))) for v in x])
        return labels, centers

    for _ in range(n_init):
        centers = rng.choice(uniq, size=k, replace=False).astype(float)
        labels = np.zeros(x.shape[0], dtype=int)
        for _ in range(max_iter):
            new_labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
            if np.array_equal(new_labels, labels) and _ > 0:
                break
            labels = new_labels
            for c in range(k):
                members = x[labels == c]
                if members.size:
                    centers[c] = np.median(members)
        cost = float(np.sum(np.abs(x - centers[labels])))
        if cost < best_cost:
            best_cost, best_labels, best_centers = cost, labels.copy(), centers.copy()
    return best_labels, best_centers


def determine_common_decay_times(t_vals, n_common_slopes, hist_resolution=0.05, seed=0):
    """Cluster decay-time estimates into common values.

    ``t_vals`` may have any shape; non-positive entries are ignored. Returns
    ``(common_times, clusters)`` with ``n_common_slopes`` sorted time values.
    """
    t = np.asarray(t_vals, dtype=float).ravel()
    nonzero = t[t > 0]
    if nonzero.size == 0:
        raise ValueError("no positive decay times to cluster")

    labels, centers = _kmedians_1d(nonzero, n_common_slopes, seed=seed)
    order = np.argsort(centers)

    edges = hist_resolution_to_edges(nonzero.min(), nonzero.max(), hist_resolution)
    common = np.zeros(n_common_slopes)
    clustered = []
    for m in range(n_common_slopes):
        members = nonzero[labels == order[m]]
        clustered.append(members)
        if members.size == 0:
            common[m] = centers[order[m]]
            continue
        counts, _ = np.histogram(members, bins=edges)
        peak = int(np.argmax(counts))
        common[m] = np.mean(edges[peak:peak + 2])
    return common, clustered
