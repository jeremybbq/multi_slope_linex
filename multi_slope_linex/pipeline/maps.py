"""Spatial maps from position-indexed values."""

from __future__ import annotations

import numpy as np


def _room_vertices(boundary):
    """Wall segments [(x0,y0,x1,y1), ...] -> vertex array (n*2, 2)."""
    b = np.asarray(boundary, dtype=float)
    return b.reshape(-1, 2)


def is_inside_boundary(pos, boundaries):
    """Test a two-dimensional position against room boundaries.

    Returns ``(inside, membership)`` where membership has one boolean per room.
    """
    pos = np.asarray(pos, dtype=float)
    inside = False
    which = np.zeros(len(boundaries), dtype=bool)
    for i, bnd in enumerate(boundaries):
        verts = _room_vertices(bnd)
        if np.all(pos >= verts.min(0)) and np.all(pos <= verts.max(0)):
            inside = True
            which[i] = True
    return inside, which


def is_on_boundary(pos, boundaries, tol=1e-6):
    """Return whether a two-dimensional position lies on a wall segment."""
    pos = np.asarray(pos, dtype=float)
    for bnd in boundaries:
        edges = np.asarray(bnd, dtype=float)
        for e in edges:
            v1, v2 = e[:2], e[2:]
            if (np.linalg.norm(v1 - pos) + np.linalg.norm(v2 - pos)
                    - np.linalg.norm(v1 - v2)) < tol:
                return True
    return False


def list2map(list_vals, list_pos, boundaries, map_res):
    """Map position-indexed values to a two-dimensional grid.

    ``list_vals`` is ``(n_positions[, n_values])`` and ``list_pos`` is
    ``(n_positions, 2)``. Returns ``(map_values, grid)``; grid points outside
    boundaries are ``NaN``.
    """
    list_vals = np.asarray(list_vals, dtype=float)
    if list_vals.ndim == 1:
        list_vals = list_vals[:, None]
    list_pos = np.asarray(list_pos, dtype=float)

    verts = np.vstack([_room_vertices(b) for b in boundaries])
    mins, maxs = verts.min(0), verts.max(0)
    x_ax = np.arange(mins[0], maxs[0] + map_res / 2, map_res)
    y_ax = np.arange(mins[1], maxs[1] + map_res / 2, map_res)
    XX, YY = np.meshgrid(x_ax, y_ax)

    n_values = list_vals.shape[1]
    map_vals = np.full((*XX.shape, n_values), np.nan)
    for r in range(XX.shape[0]):
        for c in range(XX.shape[1]):
            pos = np.array([XX[r, c], YY[r, c]])
            inside, which = is_inside_boundary(pos, boundaries)
            if not inside or is_on_boundary(pos, boundaries):
                continue
            dist = np.linalg.norm(list_pos - pos, axis=1)
            for idx in np.argsort(dist):
                _, pos_which = is_inside_boundary(list_pos[idx], boundaries)
                if np.sum(pos_which & which) >= 1:  # same room (>=1 shared membership)
                    map_vals[r, c, :] = list_vals[idx, :]
                    break
    if n_values == 1:
        map_vals = map_vals[:, :, 0]
    return map_vals, {"XX": XX, "YY": YY}
