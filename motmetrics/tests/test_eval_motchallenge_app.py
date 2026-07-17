# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Tests for the MOTChallenge command-line evaluator helpers."""

from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pytest

import motmetrics as mm
from motmetrics.apps import eval_motchallenge

DATA_DIR = Path(__file__).parents[1] / "data"
SEQUENCE_NAMES = ("TUD-Campus", "TUD-Stadtmitte")


def load_dataframes():
    ground_truths = OrderedDict()
    trackers = OrderedDict()
    for sequence_name in SEQUENCE_NAMES:
        sequence_dir = DATA_DIR / sequence_name
        ground_truths[sequence_name] = mm.io.loadtxt(sequence_dir / "gt.txt")
        trackers[sequence_name] = mm.io.loadtxt(sequence_dir / "test.txt")
    trackers["missing-ground-truth"] = trackers[SEQUENCE_NAMES[0]]
    return ground_truths, trackers


def summarize(accumulators, names):
    return mm.metrics.create().compute_many(
        accumulators,
        metrics=mm.metrics.motchallenge_metrics,
        names=names,
        generate_overall=True,
        n_jobs=1,
    )


def test_parallel_compare_dataframes_matches_serial():
    ground_truths, trackers = load_dataframes()

    serial_accumulators, serial_names = eval_motchallenge.compare_dataframes(
        ground_truths,
        trackers,
        n_jobs=1,
    )
    parallel_accumulators, parallel_names = eval_motchallenge.compare_dataframes(
        ground_truths,
        trackers,
        n_jobs=2,
    )

    assert serial_names == parallel_names == list(SEQUENCE_NAMES)
    pd.testing.assert_frame_equal(
        summarize(parallel_accumulators, parallel_names),
        summarize(serial_accumulators, serial_names),
    )


def test_compare_dataframes_rejects_invalid_worker_count():
    with pytest.raises(ValueError, match="n_jobs"):
        eval_motchallenge.compare_dataframes(OrderedDict(), OrderedDict(), n_jobs=0)
