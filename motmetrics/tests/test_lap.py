from pytest import approx
import numpy as np
import motmetrics.lap as lap


def test_lap_solvers():
    assert len(lap.available_solvers) > 0
    print(lap.available_solvers)

    costs = np.array([[6, 9, 1],[10, 3, 2],[8, 7, 4.]])
    costs_copy = costs.copy()
    results = [lap.linear_sum_assignment(costs, solver=s) for s in lap.available_solvers]
    expected = np.array([[0, 1, 2], [2, 1, 0]])
    [np.testing.assert_allclose(r, expected) for r in results]
    np.testing.assert_allclose(costs, costs_copy)


    costs = np.array([[5, 9, np.nan],[10, np.nan, 2],[8, 7, 4.]])
    costs_copy = costs.copy()
    results = [lap.linear_sum_assignment(costs, solver=s) for s in lap.available_solvers]
    expected = np.array([[0, 1, 2], [0, 2, 1]])
    [np.testing.assert_allclose(r, expected) for r in results]
    np.testing.assert_allclose(costs, costs_copy)

def test_change_solver():
    
    def mysolver(x):
        mysolver.called += 1
        return None, None
    mysolver.called = 0

    costs = np.array([[6, 9, 1],[10, 3, 2],[8, 7, 4.]])

    with lap.set_default_solver(mysolver):
        rids, cids = lap.linear_sum_assignment(costs)
    assert mysolver.called == 1
    rids, cids = lap.linear_sum_assignment(costs)
    assert mysolver.called == 1

