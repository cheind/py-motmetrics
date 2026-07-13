# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Tests for the prepared-sequence batch evaluator."""

from pathlib import Path

import numpy as np
import pandas as pd
from pytest import approx

import motmetrics as mm

DATA_DIR = Path(__file__).parents[1] / "data"
SEQUENCE_NAMES = ("TUD-Campus", "TUD-Stadtmitte")
HOTA_GOLDEN = {
    "TUD-Campus": (0.3913974378451139, 0.418047030142763, 0.36912068120832836),
    "TUD-Stadtmitte": (0.3978490169927877, 0.3922675723693166, 0.4088407518112996),
}


def load_sequences():
    return {
        name: (
            mm.io.loadtxt(DATA_DIR / name / "gt.txt"),
            mm.io.loadtxt(DATA_DIR / name / "test.txt"),
        )
        for name in SEQUENCE_NAMES
    }


def test_prepared_batch_evaluator_matches_legacy_metrics_and_events():
    sequences = load_sequences()
    batch = mm.evaluator.evaluate_many(sequences, n_jobs=1)
    host = mm.metrics.create()
    metric_names = list(
        dict.fromkeys(
            [
                *mm.metrics.motchallenge_metrics,
                "motp",
                "num_detections",
                "num_objects",
                "num_predictions",
                "idtp",
                "idfn",
                "idfp",
            ]
        )
    )

    legacy_accumulators = []
    for name, (ground_truth, tracker) in sequences.items():
        result = batch.sequences[name]
        legacy = mm.MOTAccumulator()
        for frame_id, gt_ids, tracker_ids, similarity in zip(
            result.prepared.frame_ids,
            result.prepared.gt_ids,
            result.prepared.tracker_ids,
            result.prepared.similarities,
        ):
            distances = np.where(1 - similarity > 0.5, np.nan, 1 - similarity)
            legacy.update(gt_ids, tracker_ids, distances, frameid=frame_id)
        legacy_accumulators.append(legacy)
        compact_accumulator = result.clear_identity.accumulator
        assert compact_accumulator._deferred_clear_updates is not None
        assert not compact_accumulator._indices["FrameId"]
        compact_summary = host.compute(
            compact_accumulator,
            metrics=metric_names,
            name=name,
        )
        assert compact_accumulator._deferred_clear_updates is not None
        legacy_summary = host.compute(legacy, metrics=metric_names, name=name)
        pd.testing.assert_frame_equal(
            compact_summary,
            legacy_summary,
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )
        compact_events = compact_accumulator.events
        assert compact_accumulator._deferred_clear_updates is None
        assert compact_accumulator._metric_stats is None
        pd.testing.assert_frame_equal(
            compact_events,
            legacy.events,
        )

        hota, deta, assa = HOTA_GOLDEN[name]
        assert result.hota.hota.mean() == approx(hota)
        assert result.hota.deta.mean() == approx(deta)
        assert result.hota.assa.mean() == approx(assa)

    legacy_overall = host.compute_many(
        legacy_accumulators,
        metrics=metric_names,
        names=list(sequences),
        generate_overall=True,
        n_jobs=1,
    ).loc["OVERALL"]
    for metric_name in metric_names:
        assert batch.overall_clear_identity[metric_name] == approx(
            legacy_overall[metric_name]
        )


def test_parallel_batch_evaluation_matches_serial():
    sequences = load_sequences()
    prepared = mm.evaluator.prepare_many(sequences, n_jobs=2)
    serial = mm.evaluator.evaluate_many(prepared, n_jobs=1)
    parallel = mm.evaluator.evaluate_many(prepared, n_jobs=2)

    for name in sequences:
        serial_result = serial.sequences[name]
        parallel_result = parallel.sequences[name]
        np.testing.assert_allclose(serial_result.hota.hota, parallel_result.hota.hota)
        np.testing.assert_allclose(serial_result.hota.deta, parallel_result.hota.deta)
        np.testing.assert_allclose(serial_result.hota.assa, parallel_result.hota.assa)
        for metric_name in mm.metrics.motchallenge_metrics:
            assert serial_result.clear_identity.stats[metric_name] == approx(
                parallel_result.clear_identity.stats[metric_name]
            )
