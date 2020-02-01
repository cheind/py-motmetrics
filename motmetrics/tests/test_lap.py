"""Tests linear assignment problem solvers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import motmetrics.lap as lap


def test_lap_solvers():
    """Tests that solver finds correct solution."""
    assert len(lap.available_solvers) > 0
    print(lap.available_solvers)

    costs = np.array([[6, 9, 1], [10, 3, 2], [8, 7, 4.]])
    costs_copy = costs.copy()
    results = [lap.linear_sum_assignment(costs, solver=s) for s in lap.available_solvers]
    expected = np.array([[0, 1, 2], [2, 1, 0]])
    for r in results:
        np.testing.assert_equal(r, expected)
    np.testing.assert_equal(costs, costs_copy)

    costs = np.array([[5, 9, np.nan], [10, np.nan, 2], [8, 7, 4.]])
    costs_copy = costs.copy()
    results = [lap.linear_sum_assignment(costs, solver=s) for s in lap.available_solvers]
    expected = np.array([[0, 1, 2], [0, 2, 1]])
    for r in results:
        np.testing.assert_equal(r, expected)
    np.testing.assert_equal(costs, costs_copy)


def test_change_solver():
    """Tests effect of lap.set_default_solver."""

    def mysolver(_):
        mysolver.called += 1
        return np.array([]), np.array([])
    mysolver.called = 0

    costs = np.array([[6, 9, 1], [10, 3, 2], [8, 7, 4.]])

    with lap.set_default_solver(mysolver):
        lap.linear_sum_assignment(costs)
    assert mysolver.called == 1
    lap.linear_sum_assignment(costs)
    assert mysolver.called == 1
