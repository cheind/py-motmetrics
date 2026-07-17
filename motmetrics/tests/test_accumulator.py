"""Tests for the integer-coded MOT accumulator."""

import numpy as np
import pytest

import motmetrics._metrics as metrics
from motmetrics._accumulator import _Accumulator


def _metric_values(accumulator, names):
    return metrics._METRIC_HOST.compute(accumulator, metrics=names)


def test_consecutive_matches_preserve_identity():
    accumulator = _Accumulator(np.array([2]), np.array([2]))
    accumulator.update(np.array([0]), np.array([0]), np.array([[0.1]]), np.array([[True]]))
    accumulator.update(np.array([0]), np.array([0]), np.array([[0.2]]), np.array([[True]]))

    result = _metric_values(accumulator, ["num_frames", "num_matches", "motp", "idf1"])
    assert result["num_frames"] == 2
    assert result["num_matches"] == 2
    assert result["motp"] == pytest.approx(0.15)
    assert result["idf1"] == 1.0


def test_changed_prediction_is_a_switch():
    accumulator = _Accumulator(np.array([2]), np.array([1, 1]))
    accumulator.update(np.array([0]), np.array([0]), np.array([[0.1]]), np.array([[True]]))
    accumulator.update(np.array([0]), np.array([1]), np.array([[0.1]]), np.array([[True]]))

    result = _metric_values(accumulator, ["num_matches", "num_switches"])
    assert result["num_matches"] == 1
    assert result["num_switches"] == 1


def test_empty_frames_are_counted():
    accumulator = _Accumulator(np.empty(0, dtype=int), np.empty(0, dtype=int))
    accumulator.update(
        np.empty(0, dtype=int),
        np.empty(0, dtype=int),
        np.empty((0, 0)),
        np.empty((0, 0), dtype=bool),
    )

    assert _metric_values(accumulator, ["num_frames"])["num_frames"] == 1
