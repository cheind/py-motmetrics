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


def _empty_accumulator():
    return _Accumulator(np.empty(0, dtype=int), np.empty(0, dtype=int))


def _compute(accumulator, metric_names=None):
    return metrics._compute_metrics(accumulator, metric_names=metric_names)


def _accumulate(frames):
    """Encode arbitrary test IDs once, then exercise the production path."""
    object_ids = sorted({object_id for objects, _, _ in frames for object_id in objects})
    prediction_ids = sorted({prediction_id for _, predictions, _ in frames for prediction_id in predictions})
    object_codes = {object_id: code for code, object_id in enumerate(object_ids)}
    prediction_codes = {prediction_id: code for code, prediction_id in enumerate(prediction_ids)}
    encoded_frames = []
    object_counts = np.zeros(len(object_ids), dtype=int)
    prediction_counts = np.zeros(len(prediction_ids), dtype=int)
    for objects, predictions, distances in frames:
        encoded_objects = np.asarray([object_codes[value] for value in objects], dtype=np.intp)
        encoded_predictions = np.asarray([prediction_codes[value] for value in predictions], dtype=np.intp)
        object_counts += np.bincount(encoded_objects, minlength=len(object_ids))
        prediction_counts += np.bincount(encoded_predictions, minlength=len(prediction_ids))
        encoded_frames.append(
            (
                encoded_objects,
                encoded_predictions,
                np.asarray(distances, dtype=float).reshape(len(objects), len(predictions)),
            )
        )

    accumulator = _Accumulator(object_counts, prediction_counts)
    for encoded_objects, encoded_predictions, distances in encoded_frames:
        accumulator.update(
            encoded_objects,
            encoded_predictions,
            distances,
            np.isfinite(distances),
        )
    return accumulator


def test_metrics_with_empty_state():
    acc = _empty_accumulator()

    metr = _compute(
        acc,
        metric_names=[
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
        _compute(object())


def test_accumulator_keeps_all_metrics_in_compact_state():
    acc = _accumulate([
        ([1, 2], [10, 20], [[0.1, np.nan], [np.nan, 0.2]]),
        ([1, 2], [10, 30], [[0.1, np.nan], [np.nan, 0.2]]),
    ])

    result = _compute(
        acc,
        metric_names=metrics._METRIC_SPECS,
    )

    assert result["num_frames"] == 2
    assert result["num_switches"] == 1
    assert result["idtp"] == 3


def test_assignment_metrics_with_empty_groundtruth():
    """Tests metrics when there are no ground-truth objects."""
    acc = _accumulate([
        ([], [1, 2, 3, 4], []),
        ([], [1, 2, 3, 4], []),
        ([], [1, 2, 3, 4], []),
        ([], [1, 2, 3, 4], []),
    ])

    metr = _compute(
        acc,
        metric_names=[
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
    acc = _accumulate([
        ([1, 2, 3, 4], [], []),
        ([1, 2, 3, 4], [], []),
        ([1, 2, 3, 4], [], []),
        ([1, 2, 3, 4], [], []),
    ])

    metr = _compute(
        acc,
        metric_names=[
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
    acc = _accumulate([
        ([], [], []),
        ([], [], []),
        ([], [], []),
        ([], [], []),
    ])

    metr = _compute(
        acc,
        metric_names=[
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


def test_sparse_identity_assignment_uses_independent_components():
    object_codes = np.asarray([0, 0, 1, 2])
    prediction_codes = np.asarray([0, 1, 1, 2])
    weights = np.asarray([3, 4, 5, 2])

    rows, columns, total = metrics._max_weight_matching(
        5000,
        5000,
        object_codes,
        prediction_codes,
        weights,
    )

    np.testing.assert_equal(rows, [0, 1, 2])
    np.testing.assert_equal(columns, [0, 1, 2])
    assert total == 10


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
    benchmark(
        metrics._compute_metrics,
        acc,
        metric_names=metrics._METRIC_SPECS,
    )


def _accum_random_uniform(
    rand, seq_len, num_objs, num_hyps, objs_per_frame, hyps_per_frame
):
    frames = []
    for _ in range(seq_len):
        # Choose subset of objects present in this frame.
        objs = rand.choice(num_objs, objs_per_frame, replace=False)
        # Choose subset of hypotheses present in this frame.
        hyps = rand.choice(num_hyps, hyps_per_frame, replace=False)
        dist = rand.uniform(size=(objs_per_frame, hyps_per_frame))
        frames.append((objs, hyps, dist))
    return _accumulate(frames)


def test_mota_motp():
    """Tests values of MOTA and MOTP."""
    acc = _accumulate([
        ([], [1, 2], []),
        ([1, 2], [], []),
        ([1, 2], [1, 2], [[1, 0.5], [0.3, 1]]),
        ([1, 2], [1, 2], [[0.2, np.nan], [np.nan, 0.1]]),
        ([1, 2], [1, 2], [[5, 1], [1, 5]]),
        ([], [], []),
    ])

    metr = _compute(
        acc,
        metric_names=[
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


def test_trackeval_clear_derivatives():
    acc = _accumulate([
        ([1], [1], [[0.2]]),
        ([1, 2], [1], [[0.1], [np.nan]]),
        ([], [2], []),
    ])

    result = _compute(
        acc,
        metric_names=[
            "moda",
            "smota",
            "mtr",
            "ptr",
            "mlr",
            "clr_f1",
            "fp_per_frame",
        ],
    )

    assert result["moda"] == approx(1 / 3)
    assert result["smota"] == approx(0.7 / 3)
    assert result["mtr"] == approx(0.5)
    assert result["ptr"] == approx(0)
    assert result["mlr"] == approx(0.5)
    assert result["clr_f1"] == approx(2 / 3)
    assert result["fp_per_frame"] == approx(1 / 3)


def test_identity_change_metrics():
    acc = _accumulate([
        ([], [], []),
        ([1, 2], [1, 2], [[1, 0], [0, 1]]),
        ([1, 2], [1, 2], [[0.4, np.nan], [np.nan, 0.4]]),
        ([1, 2], [1, 2], [[0, 1], [1, 0]]),
        ([1, 2], [2, 3], [[1, 0], [0.4, 0.7]]),
        ([1, 3], [2, 3], [[1, 0], [0.4, 0.7]]),
        ([], [], []),
    ])

    metr = _compute(
        acc,
        metric_names=[
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
    acc = _accumulate([
        ([1, 2, 3, 4], [], []),
        ([1, 2, 3, 4], [], []),
        ([1, 2, 3, 4], [], []),
        ([1, 2, 3, 4], [], []),
        ([4], [4], [0]),
        ([4], [4], [0]),
        ([4], [4], [0]),
        ([4], [4], [0]),
    ])

    metr = _compute(acc, metric_names="mota")
    assert metr["mota"] == approx(0.2)


def test_track_quality_boundary_matches_trackeval():
    acc = _accumulate([
        ([1], [1], [0]),
        ([1], [1], [0]),
        ([1], [1], [0]),
        ([1], [1], [0]),
        ([1], [], []),
    ])

    metr = _compute(
        acc,
        metric_names=["mostly_tracked", "partially_tracked", "mostly_lost"],
    )
    assert metr["mostly_tracked"] == 0
    assert metr["partially_tracked"] == 1
    assert metr["mostly_lost"] == 0


def test_num_fragmentations_ignores_leading_and_trailing_misses():
    acc = _accumulate([
        ([1, 2], [], []),
        ([1, 2], [1], [[0.1], [np.nan]]),
        ([1, 2], [], []),
        ([1, 2], [1], [[0.1], [np.nan]]),
        ([1, 2], [], []),
    ])

    summary = _compute(acc, metric_names=["num_fragmentations"])

    assert summary['num_fragmentations'] == 1
