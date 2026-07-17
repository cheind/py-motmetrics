"""Single-backend linear assignment for the evaluation hot path."""

import numpy as np
from scipy.optimize import linear_sum_assignment as _scipy_assignment


def _linear_sum_assignment(costs):
    """Solve a possibly sparse rectangular assignment with SciPy."""
    costs = np.asarray(costs)
    if costs.size == 0:
        empty = np.empty(0, dtype=int)
        return empty, empty

    finite = np.isfinite(costs)
    if finite.all():
        return _scipy_assignment(costs)
    if not finite.any():
        empty = np.empty(0, dtype=int)
        return empty, empty

    max_pairs = min(costs.shape)
    max_abs_cost = np.abs(costs[finite]).max()
    forbidden_cost = 2 * max_pairs * (max_abs_cost + 1) + 1
    row_indices, column_indices = _scipy_assignment(
        np.where(finite, costs, forbidden_cost)
    )
    valid = finite[row_indices, column_indices]
    return row_indices[valid], column_indices[valid]
