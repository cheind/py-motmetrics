"""Single-backend linear assignment for the evaluation hot path."""

import numpy as np
from lap import lapjv as _lapjv


def _dense_linear_sum_assignment(costs):
    """Solve a finite rectangular assignment with LapX."""
    costs = np.asarray(costs)
    if costs.size == 0:
        empty = np.empty(0, dtype=int)
        return empty, empty

    row_to_column, _ = _lapjv(
        costs,
        extend_cost=costs.shape[0] != costs.shape[1],
        return_cost=False,
    )
    row_indices = np.flatnonzero(row_to_column >= 0)
    return row_indices, row_to_column[row_indices].astype(int, copy=False)


def _linear_sum_assignment(costs, finite=None):
    """Solve a possibly forbidden-edge rectangular assignment with LapX."""
    costs = np.asarray(costs)
    if costs.size == 0:
        empty = np.empty(0, dtype=int)
        return empty, empty

    if finite is None:
        finite = np.isfinite(costs)
    if finite.all():
        return _dense_linear_sum_assignment(costs)
    if not finite.any():
        empty = np.empty(0, dtype=int)
        return empty, empty

    max_pairs = min(costs.shape)
    max_abs_cost = np.abs(costs[finite]).max()
    forbidden_cost = 2 * max_pairs * (max_abs_cost + 1) + 1
    row_indices, column_indices = _dense_linear_sum_assignment(
        np.where(finite, costs, forbidden_cost)
    )
    valid = finite[row_indices, column_indices]
    return row_indices[valid], column_indices[valid]
