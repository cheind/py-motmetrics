"""Tests for the private state-only accumulator."""

import pytest

import motmetrics._metrics as metrics
from motmetrics._accumulator import _Accumulator


def _metric_values(accumulator, names):
    return metrics._METRIC_HOST.compute(
        accumulator,
        metrics=names,
    )


def test_auto_frame_ids():
    accumulator = _Accumulator(auto_id=True)

    assert accumulator.update([1], [10], [[0.1]]) == 0
    assert accumulator.update([1], [10], [[0.2]]) == 1
    with pytest.raises(AssertionError, match="Cannot provide frame id"):
        accumulator.update([], [], [], frameid=2)

    result = _metric_values(accumulator, ["num_frames", "num_matches", "motp", "idf1"])
    assert result["num_frames"] == 2
    assert result["num_matches"] == 2
    assert result["motp"] == pytest.approx(0.15)
    assert result["idf1"] == 1.0


def test_manual_frame_ids_are_required_by_default():
    accumulator = _Accumulator()

    assert accumulator.update([], [], [], frameid=7) == 7
    with pytest.raises(AssertionError, match="auto-id is not enabled"):
        accumulator.update([], [], [])


def test_max_switch_time_limits_switches():
    near = _Accumulator(max_switch_time=1)
    near.update([1], [1], [[0.1]], frameid=1)
    near.update([1], [2], [[0.1]], frameid=2)

    far = _Accumulator(max_switch_time=1)
    far.update([1], [1], [[0.1]], frameid=1)
    far.update([1], [2], [[0.1]], frameid=5)

    assert _metric_values(near, ["num_switches"])["num_switches"] == 1
    assert _metric_values(far, ["num_switches"])["num_switches"] == 0


def test_reset_discards_all_metric_state():
    accumulator = _Accumulator(auto_id=True)
    accumulator.update([1], [1], [[0.1]])
    accumulator.reset()

    result = _metric_values(accumulator, ["num_frames", "num_objects", "num_predictions"])
    assert result["num_frames"] == 0
    assert result["num_objects"] == 0
    assert result["num_predictions"] == 0
