from pytest import approx
import numpy as np
from motmetrics.lap import linear_sum_assignment as lsa, available_solvers

all_solvers = available_solvers

def test_lap_solvers():
    assert len(all_solvers) > 0

    costs = np.array([[6, 9, 1],[10, 3, 2],[8, 7, 4.]])
    costs_copy = costs.copy()
    results = [lsa(costs, solver=s) for s in all_solvers]
    expected = np.array([[0, 1, 2], [2, 1, 0]])
    [np.testing.assert_allclose(r, expected) for r in results]
    np.testing.assert_allclose(costs, costs_copy)


    costs = np.array([[5, 9, np.nan],[10, np.nan, 2],[8, 7, 4.]])
    costs_copy = costs.copy()
    results = [lsa(costs, solver=s) for s in all_solvers]
    expected = np.array([[0, 1, 2], [0, 2, 1]])
    [np.testing.assert_allclose(r, expected) for r in results]
    np.testing.assert_allclose(costs, costs_copy)

