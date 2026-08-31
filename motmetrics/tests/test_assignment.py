"""Tests for the single LapX assignment path."""

import numpy as np
import pytest

from motmetrics._assignment import _linear_sum_assignment


@pytest.mark.parametrize(
    ("costs", "expected"),
    [
        (
            [[6, 9, 1], [10, 3, 2], [8, 7, 4]],
            [[0, 1, 2], [2, 1, 0]],
        ),
        (
            [[5, 5, 6], [1, 2, 5], [2, 4, 5]],
            [[0, 1, 2], [2, 1, 0]],
        ),
        (
            [[np.nan, np.nan, 2], [np.nan, np.nan, 1], [8, 7, 4]],
            [[1, 2], [2, 1]],
        ),
        (
            [[5, 9, np.nan], [10, np.nan, 2], [8, 7, 4]],
            [[0, 1, 2], [0, 2, 1]],
        ),
    ],
)
def test_assignment(costs, expected):
    costs = np.asarray(costs, dtype=float)
    original = costs.copy()

    result = _linear_sum_assignment(costs)

    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, original)


@pytest.mark.parametrize("shape", [(0, 0), (1, 0), (0, 1)])
def test_empty_assignment(shape):
    rows, columns = _linear_sum_assignment(np.empty(shape))

    assert rows.size == 0
    assert columns.size == 0


def test_rectangular_assignment_excludes_forbidden_edges():
    costs = np.asarray([
        [1, np.nan, np.nan],
        [np.nan, 1, np.nan],
    ])

    rows, columns = _linear_sum_assignment(costs)

    np.testing.assert_equal(rows, [0, 1])
    np.testing.assert_equal(columns, [0, 1])


def test_all_forbidden_edges_return_no_matches():
    rows, columns = _linear_sum_assignment(np.full((3, 2), np.nan))

    assert rows.size == 0
    assert columns.size == 0
