# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Tests computation of metrics from accumulator."""

import numpy as np
from pytest import approx, raises

import motmetrics._metrics as metrics
from motmetrics._accumulator import _Accumulator


def test_metricscontainer_1():
    """Tests registration of events with dependencies."""
    m = metrics._MetricsHost()
    m._register(lambda engine: 1.0, name="a")
    m._register(lambda engine: 2.0, name="b")
    m._register(lambda engine, a, b: a + b, deps=["a", "b"], name="add")
    m._register(lambda engine, a, b: a - b, deps=["a", "b"], name="sub")
    m._register(lambda engine, a, b: a * b, deps=["add", "sub"], name="mul")
    summary = m.compute(_Accumulator(), metrics=["mul", "add"])
    assert summary["mul"] == -3.0
    assert summary["add"] == 3.0


def test_metricscontainer_autodep():
    """Tests automatic dependencies from argument names."""
    m = metrics._MetricsHost()
    m._register(lambda engine: 1.0, name="a")
    m._register(lambda engine: 2.0, name="b")
    m._register(lambda engine, a, b: a + b, name="add", deps="auto")
    m._register(lambda engine, a, b: a - b, name="sub", deps="auto")
    m._register(lambda engine, add, sub: add * sub, name="mul", deps="auto")
    summary = m.compute(_Accumulator(), metrics=["mul", "add"])
    assert summary["mul"] == -3.0
    assert summary["add"] == 3.0


def test_metricscontainer_autoname():
    """Tests automatic names (and dependencies) from inspection."""

    def constant_a(_):
        """Constant a help."""
        return 1.0

    def constant_b(_):
        return 2.0

    def add(_, constant_a, constant_b):
        return constant_a + constant_b

    def sub(_, constant_a, constant_b):
        return constant_a - constant_b

    def mul(_, add, sub):
        return add * sub

    m = metrics._MetricsHost()
    m._register(constant_a, deps="auto")
    m._register(constant_b, deps="auto")
    m._register(add, deps="auto")
    m._register(sub, deps="auto")
    m._register(mul, deps="auto")

    summary = m.compute(_Accumulator(), metrics=["mul", "add"])
    assert summary["mul"] == -3.0
    assert summary["add"] == 3.0


def test_metrics_with_empty_state():
    acc = _Accumulator()

    mh = metrics._METRIC_HOST
    metr = mh.compute(
        acc,
        metrics=[
            "mota",
            "motp",
            "num_predictions",
            "num_objects",
            "num_detections",
            "num_frames",
        ],
    )
    assert np.isnan(metr["mota"])
    assert np.isnan(metr["motp"])
    assert metr["num_predictions"] == 0
    assert metr["num_objects"] == 0
    assert metr["num_detections"] == 0
    assert metr["num_frames"] == 0


def test_metric_input_must_be_the_compact_accumulator():
    with raises(TypeError, match="requires an accumulator"):
        metrics._METRIC_HOST.compute(object())


def test_accumulator_keeps_all_metrics_in_compact_state():
    acc = _Accumulator(auto_id=True)
    acc.update([1, 2], [10, 20], [[0.1, np.nan], [np.nan, 0.2]])
    acc.update([1, 2], [10, 30], [[0.1, np.nan], [np.nan, 0.2]])

    metric_host = metrics._METRIC_HOST
    result = metric_host.compute(
        acc,
        metrics=metric_host.names,
    )

    assert result["num_frames"] == 2
    assert result["num_switches"] == 1
    assert result["idtp"] == 3


def test_assignment_metrics_with_empty_groundtruth():
    """Tests metrics when there are no ground-truth objects."""
    acc = _Accumulator(auto_id=True)
    # Empty groundtruth.
    acc.update([], [1, 2, 3, 4], [])
    acc.update([], [1, 2, 3, 4], [])
    acc.update([], [1, 2, 3, 4], [])
    acc.update([], [1, 2, 3, 4], [])

    mh = metrics._METRIC_HOST
    metr = mh.compute(
        acc,
        metrics=[
            "num_matches",
            "num_false_positives",
            "num_misses",
            "idtp",
            "idfp",
            "idfn",
            "num_frames",
        ],
    )
    assert metr["num_matches"] == 0
    assert metr["num_false_positives"] == 16
    assert metr["num_misses"] == 0
    assert metr["idtp"] == 0
    assert metr["idfp"] == 16
    assert metr["idfn"] == 0
    assert metr["num_frames"] == 4


def test_assignment_metrics_with_empty_predictions():
    """Tests metrics when there are no predictions."""
    acc = _Accumulator(auto_id=True)
    # Empty predictions.
    acc.update([1, 2, 3, 4], [], [])
    acc.update([1, 2, 3, 4], [], [])
    acc.update([1, 2, 3, 4], [], [])
    acc.update([1, 2, 3, 4], [], [])

    mh = metrics._METRIC_HOST
    metr = mh.compute(
        acc,
        metrics=[
            "num_matches",
            "num_false_positives",
            "num_misses",
            "idtp",
            "idfp",
            "idfn",
            "num_frames",
        ],
    )
    assert metr["num_matches"] == 0
    assert metr["num_false_positives"] == 0
    assert metr["num_misses"] == 16
    assert metr["idtp"] == 0
    assert metr["idfp"] == 0
    assert metr["idfn"] == 16
    assert metr["num_frames"] == 4


def test_assignment_metrics_with_both_empty():
    """Tests metrics when there are no ground-truth objects or predictions."""
    acc = _Accumulator(auto_id=True)
    # Empty groundtruth and empty predictions.
    acc.update([], [], [])
    acc.update([], [], [])
    acc.update([], [], [])
    acc.update([], [], [])

    mh = metrics._METRIC_HOST
    metr = mh.compute(
        acc,
        metrics=[
            "num_matches",
            "num_false_positives",
            "num_misses",
            "idtp",
            "idfp",
            "idfn",
            "num_frames",
        ],
    )
    assert metr["num_matches"] == 0
    assert metr["num_false_positives"] == 0
    assert metr["num_misses"] == 0
    assert metr["idtp"] == 0
    assert metr["idfp"] == 0
    assert metr["idfn"] == 0
    assert metr["num_frames"] == 4


def test_benchmark_all_metrics(benchmark):
    """Benchmarks the only supported, accumulator-native metric path."""
    rand = np.random.RandomState(0)
    acc = _accum_random_uniform(
        rand,
        seq_len=100,
        num_objs=50,
        num_hyps=5000,
        objs_per_frame=20,
        hyps_per_frame=40,
    )
    metric_host = metrics._METRIC_HOST
    benchmark(
        metric_host.compute,
        acc,
        metrics=metric_host.names,
    )


def _accum_random_uniform(
    rand, seq_len, num_objs, num_hyps, objs_per_frame, hyps_per_frame
):
    acc = _Accumulator(auto_id=True)
    for _ in range(seq_len):
        # Choose subset of objects present in this frame.
        objs = rand.choice(num_objs, objs_per_frame, replace=False)
        # Choose subset of hypotheses present in this frame.
        hyps = rand.choice(num_hyps, hyps_per_frame, replace=False)
        dist = rand.uniform(size=(objs_per_frame, hyps_per_frame))
        acc.update(objs, hyps, dist)
    return acc


def test_mota_motp():
    """Tests values of MOTA and MOTP."""
    acc = _Accumulator()

    # All FP
    acc.update([], [1, 2], [], frameid=0)
    # All miss
    acc.update([1, 2], [], [], frameid=1)
    # Match
    acc.update([1, 2], [1, 2], [[1, 0.5], [0.3, 1]], frameid=2)
    # Switch
    acc.update([1, 2], [1, 2], [[0.2, np.nan], [np.nan, 0.1]], frameid=3)
    # Match. Better new match is available but should prefer history
    acc.update([1, 2], [1, 2], [[5, 1], [1, 5]], frameid=4)
    # No data
    acc.update([], [], [], frameid=5)

    mh = metrics._METRIC_HOST
    metr = mh.compute(
        acc,
        metrics=[
            "num_matches",
            "num_false_positives",
            "num_misses",
            "num_switches",
            "num_detections",
            "num_objects",
            "num_predictions",
            "mota",
            "motp",
            "num_frames",
        ],
    )

    assert metr["num_matches"] == 4
    assert metr["num_false_positives"] == 2
    assert metr["num_misses"] == 2
    assert metr["num_switches"] == 2
    assert metr["num_detections"] == 6
    assert metr["num_objects"] == 8
    assert metr["num_predictions"] == 8
    assert metr["mota"] == approx(1.0 - (2 + 2 + 2) / 8)
    assert metr["motp"] == approx(11.1 / 6)
    assert metr["num_frames"] == 6


def test_ids():
    """Test metrics with frame IDs specified manually."""
    acc = _Accumulator()

    # No data
    acc.update([], [], [], frameid=0)
    # Match
    acc.update([1, 2], [1, 2], [[1, 0], [0, 1]], frameid=1)
    # Switch also Transfer
    acc.update([1, 2], [1, 2], [[0.4, np.nan], [np.nan, 0.4]], frameid=2)
    # Match
    acc.update([1, 2], [1, 2], [[0, 1], [1, 0]], frameid=3)
    # Ascend (switch)
    acc.update([1, 2], [2, 3], [[1, 0], [0.4, 0.7]], frameid=4)
    # Migrate (transfer)
    acc.update([1, 3], [2, 3], [[1, 0], [0.4, 0.7]], frameid=5)
    # No data
    acc.update([], [], [], frameid=6)

    mh = metrics._METRIC_HOST
    metr = mh.compute(
        acc,
        metrics=[
            "num_matches",
            "num_false_positives",
            "num_misses",
            "num_switches",
            "num_transfer",
            "num_ascend",
            "num_migrate",
            "num_detections",
            "num_objects",
            "num_predictions",
            "mota",
            "motp",
            "num_frames",
        ],
    )
    assert metr["num_matches"] == 7
    assert metr["num_false_positives"] == 0
    assert metr["num_misses"] == 0
    assert metr["num_switches"] == 3
    assert metr["num_transfer"] == 3
    assert metr["num_ascend"] == 1
    assert metr["num_migrate"] == 1
    assert metr["num_detections"] == 10
    assert metr["num_objects"] == 10
    assert metr["num_predictions"] == 10
    assert metr["mota"] == approx(1.0 - (0 + 0 + 3) / 10)
    assert metr["motp"] == approx(1.6 / 10)
    assert metr["num_frames"] == 7


def test_correct_average():
    """Tests what is depicted in figure 3 of 'Evaluating MOT Performance'."""
    acc = _Accumulator(auto_id=True)

    # No track
    acc.update([1, 2, 3, 4], [], [])
    acc.update([1, 2, 3, 4], [], [])
    acc.update([1, 2, 3, 4], [], [])
    acc.update([1, 2, 3, 4], [], [])

    # Track single
    acc.update([4], [4], [0])
    acc.update([4], [4], [0])
    acc.update([4], [4], [0])
    acc.update([4], [4], [0])

    mh = metrics._METRIC_HOST
    metr = mh.compute(acc, metrics="mota")
    assert metr["mota"] == approx(0.2)


def test_track_quality_boundary_matches_trackeval():
    acc = _Accumulator(auto_id=True)

    acc.update([1], [1], [0])
    acc.update([1], [1], [0])
    acc.update([1], [1], [0])
    acc.update([1], [1], [0])
    acc.update([1], [], [])

    mh = metrics._METRIC_HOST
    metr = mh.compute(
        acc,
        metrics=["mostly_tracked", "partially_tracked", "mostly_lost"],
    )
    assert metr["mostly_tracked"] == 0
    assert metr["partially_tracked"] == 1
    assert metr["mostly_lost"] == 0


def test_num_fragmentations_ignores_leading_and_trailing_misses():
    acc = _Accumulator(auto_id=True)
    acc.update([1, 2], [], [])
    acc.update([1, 2], [1], [[0.1], [np.nan]])
    acc.update([1, 2], [], [])
    acc.update([1, 2], [1], [[0.1], [np.nan]])
    acc.update([1, 2], [], [])

    summary = metrics._METRIC_HOST.compute(acc, metrics=['num_fragmentations'])

    assert summary['num_fragmentations'] == 1
