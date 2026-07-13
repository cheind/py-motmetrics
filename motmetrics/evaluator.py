# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Fast batch evaluation on a shared, array-based sequence representation."""

from __future__ import absolute_import, division, print_function

import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from motmetrics import math_util
from motmetrics.distances import iou_matrix
from motmetrics.lap import linear_sum_assignment
from motmetrics.mot import MOTAccumulator

DEFAULT_HOTA_THRESHOLDS = np.arange(0.05, 0.99, 0.05)
DEFAULT_BOX_COLUMNS = ("X", "Y", "Width", "Height")


@dataclass(frozen=True)
class PreparedSequence:
    """Frame arrays and IoU matrices shared by every metric family."""

    frame_ids: tuple
    gt_ids: tuple
    tracker_ids: tuple
    gt_id_codes: tuple
    tracker_id_codes: tuple
    similarities: tuple
    unique_gt_ids: np.ndarray
    unique_tracker_ids: np.ndarray
    gt_id_map: dict
    tracker_id_map: dict
    gt_counts: np.ndarray
    tracker_counts: np.ndarray

    @property
    def num_gt_detections(self):
        return int(self.gt_counts.sum())

    @property
    def num_tracker_detections(self):
        return int(self.tracker_counts.sum())


@dataclass
class HOTAResult:
    """HOTA-family results for all alpha thresholds."""

    thresholds: np.ndarray
    hota: np.ndarray
    deta: np.ndarray
    assa: np.ndarray
    num_detections: np.ndarray
    num_objects: int
    num_predictions: int
    deferred_frames: list = field(default=None, repr=False)

    def to_accumulators(self):
        """Return legacy threshold-specific accumulators with lazy events."""
        if self.deferred_frames is None:
            raise ValueError("Combined HOTA results do not have sequence events")

        accumulators = []
        for index, threshold in enumerate(self.thresholds):
            detections = int(self.num_detections[index])
            false_positives = self.num_predictions - detections
            stats = {
                "num_detections": detections,
                "num_objects": self.num_objects,
                "num_false_positives": false_positives,
                "deta_alpha": float(self.deta[index]),
                "assa_alpha": float(self.assa[index]),
                "hota_alpha": float(self.hota[index]),
            }
            accumulator = MOTAccumulator()
            accumulator._defer_hota_event_updates(
                self.deferred_frames,
                threshold,
                stats,
            )
            accumulators.append(accumulator)
        return accumulators

    @classmethod
    def combine(cls, results):
        """Combine sequences using TrackEval's count-weighted rules."""
        results = list(results)
        if not results:
            raise ValueError("At least one HOTA result is required")
        thresholds = results[0].thresholds
        if any(not np.array_equal(result.thresholds, thresholds) for result in results[1:]):
            raise ValueError("HOTA thresholds must match across sequences")

        detections = np.sum([result.num_detections for result in results], axis=0)
        num_objects = sum(result.num_objects for result in results)
        num_predictions = sum(result.num_predictions for result in results)
        false_positives = num_predictions - detections
        deta = detections / np.maximum(1, num_objects + false_positives)
        weighted_assa = np.sum(
            [result.assa * result.num_detections for result in results],
            axis=0,
        )
        assa = math_util.quiet_divide(weighted_assa, np.maximum(1, detections))
        hota = np.sqrt(deta * assa)
        return cls(
            thresholds=thresholds.copy(),
            hota=hota,
            deta=deta,
            assa=assa,
            num_detections=detections,
            num_objects=num_objects,
            num_predictions=num_predictions,
        )


@dataclass
class ClearIdentityResult:
    """Compact CLEAR and Identity statistics with a legacy accumulator."""

    stats: dict
    accumulator: MOTAccumulator


@dataclass
class SequenceResult:
    """All metric families for one prepared sequence."""

    prepared: PreparedSequence
    clear_identity: ClearIdentityResult
    hota: HOTAResult


@dataclass
class BatchResult:
    """Per-sequence and combined results returned by :func:`evaluate_many`."""

    sequences: OrderedDict
    overall_clear_identity: dict
    overall_hota: HOTAResult


def prepare_sequence(ground_truth, tracker, box_columns=DEFAULT_BOX_COLUMNS):
    """Convert MOT DataFrames into reusable per-frame arrays and IoU matrices."""
    box_columns = list(box_columns)
    ground_truth = ground_truth[box_columns]
    tracker = tracker[box_columns]
    frame_ids = tuple(ground_truth.index.union(tracker.index).levels[0])
    ground_truth_frames = dict(iter(ground_truth.groupby("FrameId")))
    tracker_frames = dict(iter(tracker.groupby("FrameId")))

    unique_gt_ids = ground_truth.index.get_level_values("Id").unique().to_numpy()
    unique_tracker_ids = tracker.index.get_level_values("Id").unique().to_numpy()
    gt_id_map = {object_id: index for index, object_id in enumerate(unique_gt_ids)}
    tracker_id_map = {
        tracker_id: index for index, tracker_id in enumerate(unique_tracker_ids)
    }

    gt_ids_by_frame = []
    tracker_ids_by_frame = []
    gt_codes_by_frame = []
    tracker_codes_by_frame = []
    similarities = []
    gt_counts = np.zeros(len(unique_gt_ids), dtype=np.int64)
    tracker_counts = np.zeros(len(unique_tracker_ids), dtype=np.int64)

    for frame_id in frame_ids:
        ground_truth_frame = ground_truth_frames.get(frame_id)
        tracker_frame = tracker_frames.get(frame_id)
        gt_ids = (
            ground_truth_frame.index.get_level_values("Id").to_numpy()
            if ground_truth_frame is not None
            else np.empty(0, dtype=unique_gt_ids.dtype)
        )
        tracker_ids = (
            tracker_frame.index.get_level_values("Id").to_numpy()
            if tracker_frame is not None
            else np.empty(0, dtype=unique_tracker_ids.dtype)
        )
        gt_codes = np.fromiter((gt_id_map[object_id] for object_id in gt_ids), dtype=int)
        tracker_codes = np.fromiter(
            (tracker_id_map[tracker_id] for tracker_id in tracker_ids),
            dtype=int,
        )
        np.add.at(gt_counts, gt_codes, 1)
        np.add.at(tracker_counts, tracker_codes, 1)

        if gt_ids.size and tracker_ids.size:
            similarity = iou_matrix(
                ground_truth_frame.to_numpy(dtype=float, copy=False),
                tracker_frame.to_numpy(dtype=float, copy=False),
                return_dist=False,
            )
        else:
            similarity = np.empty((len(gt_ids), len(tracker_ids)), dtype=float)

        gt_ids_by_frame.append(gt_ids)
        tracker_ids_by_frame.append(tracker_ids)
        gt_codes_by_frame.append(gt_codes)
        tracker_codes_by_frame.append(tracker_codes)
        similarities.append(similarity)

    return PreparedSequence(
        frame_ids=frame_ids,
        gt_ids=tuple(gt_ids_by_frame),
        tracker_ids=tuple(tracker_ids_by_frame),
        gt_id_codes=tuple(gt_codes_by_frame),
        tracker_id_codes=tuple(tracker_codes_by_frame),
        similarities=tuple(similarities),
        unique_gt_ids=unique_gt_ids,
        unique_tracker_ids=unique_tracker_ids,
        gt_id_map=gt_id_map,
        tracker_id_map=tracker_id_map,
        gt_counts=gt_counts,
        tracker_counts=tracker_counts,
    )


def compute_hota(prepared, thresholds=DEFAULT_HOTA_THRESHOLDS):
    """Compute every HOTA alpha from a prepared sequence."""
    thresholds = np.atleast_1d(np.asarray(thresholds, dtype=float))
    num_gt_ids = len(prepared.unique_gt_ids)
    num_tracker_ids = len(prepared.unique_tracker_ids)
    potential_matches = np.zeros((num_gt_ids, num_tracker_ids), dtype=float)

    for gt_codes, tracker_codes, similarity in zip(
        prepared.gt_id_codes,
        prepared.tracker_id_codes,
        prepared.similarities,
    ):
        if not gt_codes.size or not tracker_codes.size:
            continue
        denominator = (
            similarity.sum(axis=0)[None, :]
            + similarity.sum(axis=1)[:, None]
            - similarity
        )
        similarity_iou = np.zeros_like(similarity)
        valid = denominator > np.finfo(float).eps
        similarity_iou[valid] = similarity[valid] / denominator[valid]
        potential_matches[gt_codes[:, None], tracker_codes[None, :]] += similarity_iou

    global_alignment = potential_matches / np.maximum(
        1,
        prepared.gt_counts[:, None]
        + prepared.tracker_counts[None, :]
        - potential_matches,
    )
    match_counts = np.zeros(
        (len(thresholds), num_gt_ids, num_tracker_ids),
        dtype=np.int64,
    )
    num_detections = np.zeros(len(thresholds), dtype=np.int64)
    deferred_frames = []

    for frame_id, gt_ids, tracker_ids, gt_codes, tracker_codes, similarity in zip(
        prepared.frame_ids,
        prepared.gt_ids,
        prepared.tracker_ids,
        prepared.gt_id_codes,
        prepared.tracker_id_codes,
        prepared.similarities,
    ):
        weighted_similarity = np.empty_like(similarity)
        if gt_codes.size and tracker_codes.size:
            weighted_similarity = (
                similarity
                * global_alignment[gt_codes[:, None], tracker_codes[None, :]]
            )
        assignment = linear_sum_assignment(1 - weighted_similarity)
        deferred_frames.append((gt_ids, tracker_ids, frame_id, similarity, assignment))

        rows, columns = assignment
        if not rows.size:
            continue
        assigned_similarity = similarity[rows, columns]
        threshold_indices, match_indices = np.where(
            assigned_similarity[None, :]
            >= thresholds[:, None] - np.finfo(float).eps
        )
        np.add.at(
            match_counts,
            (
                threshold_indices,
                gt_codes[rows[match_indices]],
                tracker_codes[columns[match_indices]],
            ),
            1,
        )
        num_detections += np.bincount(
            threshold_indices,
            minlength=len(thresholds),
        )

    deta = num_detections / np.maximum(
        1,
        prepared.num_gt_detections
        + prepared.num_tracker_detections
        - num_detections,
    )
    assa = np.empty(len(thresholds), dtype=float)
    for index, counts in enumerate(match_counts):
        association = counts / np.maximum(
            1,
            prepared.gt_counts[:, None]
            + prepared.tracker_counts[None, :]
            - counts,
        )
        assa[index] = (association * counts).sum() / max(1, num_detections[index])

    return HOTAResult(
        thresholds=thresholds,
        hota=np.sqrt(deta * assa),
        deta=deta,
        assa=assa,
        num_detections=num_detections,
        num_objects=prepared.num_gt_detections,
        num_predictions=prepared.num_tracker_detections,
        deferred_frames=deferred_frames,
    )


def _identity_assignment(prepared, distance_threshold):
    pair_counts = np.zeros(
        (len(prepared.unique_gt_ids), len(prepared.unique_tracker_ids)),
        dtype=np.int64,
    )
    for gt_codes, tracker_codes, similarity in zip(
        prepared.gt_id_codes,
        prepared.tracker_id_codes,
        prepared.similarities,
    ):
        if not gt_codes.size or not tracker_codes.size:
            continue
        rows, columns = np.where((1 - similarity) <= distance_threshold)
        np.add.at(pair_counts, (gt_codes[rows], tracker_codes[columns]), 1)

    num_gt_ids, num_tracker_ids = pair_counts.shape
    size = num_gt_ids + num_tracker_ids
    false_positive_matrix = np.zeros((size, size), dtype=float)
    false_negative_matrix = np.zeros((size, size), dtype=float)
    false_positive_matrix[num_gt_ids:, :num_tracker_ids] = np.nan
    false_negative_matrix[:num_gt_ids, num_tracker_ids:] = np.nan
    false_negative_matrix[:num_gt_ids, :num_tracker_ids] = prepared.gt_counts[:, None]
    false_positive_matrix[:num_gt_ids, :num_tracker_ids] = prepared.tracker_counts[None, :]
    false_negative_matrix[
        np.arange(num_gt_ids), num_tracker_ids + np.arange(num_gt_ids)
    ] = prepared.gt_counts
    false_positive_matrix[
        num_gt_ids + np.arange(num_tracker_ids), np.arange(num_tracker_ids)
    ] = prepared.tracker_counts
    false_positive_matrix[:num_gt_ids, :num_tracker_ids] -= pair_counts
    false_negative_matrix[:num_gt_ids, :num_tracker_ids] -= pair_counts
    costs = false_positive_matrix + false_negative_matrix
    rows, columns = linear_sum_assignment(costs)
    return {
        "fpmatrix": false_positive_matrix,
        "fnmatrix": false_negative_matrix,
        "rids": rows,
        "cids": columns,
        "costs": costs,
        "min_cost": costs[rows, columns].sum(),
    }


def _compute_clear_statistics(prepared, distance_threshold):
    """Run CLEAR matching without constructing per-event Python objects."""
    object_to_hypothesis = {}
    hypothesis_to_object = {}
    last_occurrence = {}
    last_match = {}
    hypothesis_history = {}
    last_update_frame_id = None
    type_counts = {
        event_type: 0
        for event_type in (
            "MATCH",
            "SWITCH",
            "TRANSFER",
            "ASCEND",
            "MIGRATE",
            "MISS",
            "FP",
        )
    }
    tracked_counts = np.zeros(len(prepared.unique_gt_ids), dtype=np.int64)
    was_tracked = np.zeros(len(prepared.unique_gt_ids), dtype=bool)
    missed_after_track = np.zeros(len(prepared.unique_gt_ids), dtype=bool)
    num_fragmentations = 0
    distance_sum = 0.0

    def record_detection(object_code, distance, event_type):
        nonlocal distance_sum, num_fragmentations
        type_counts[event_type] += 1
        tracked_counts[object_code] += 1
        distance_sum += distance
        if missed_after_track[object_code]:
            num_fragmentations += 1
            missed_after_track[object_code] = False
        was_tracked[object_code] = True

    for frame_id, gt_ids, tracker_ids, gt_codes, similarity in zip(
        prepared.frame_ids,
        prepared.gt_ids,
        prepared.tracker_ids,
        prepared.gt_id_codes,
        prepared.similarities,
    ):
        distances = 1 - similarity
        distances = np.where(distances > distance_threshold, np.nan, distances)
        costs_for_matching = distances.copy()
        gt_masked = np.zeros(len(gt_ids), dtype=bool)
        tracker_masked = np.zeros(len(tracker_ids), dtype=bool)

        if len(gt_ids) and len(tracker_ids):
            # Preserve MOTAccumulator's continuity rule before solving the
            # remaining frame assignment.
            for gt_index, object_id in enumerate(gt_ids):
                if not (
                    object_id in object_to_hypothesis
                    and last_match[object_id] == last_update_frame_id
                ):
                    continue
                previous_hypothesis = object_to_hypothesis[object_id]
                tracker_indices = np.flatnonzero(
                    ~tracker_masked & (tracker_ids == previous_hypothesis)
                )
                if not tracker_indices.size:
                    continue
                tracker_index = tracker_indices[0]
                distance = distances[gt_index, tracker_index]
                if not np.isfinite(distance):
                    continue
                gt_masked[gt_index] = True
                tracker_masked[tracker_index] = True
                object_to_hypothesis[object_id] = previous_hypothesis
                last_match[object_id] = frame_id
                hypothesis_history[previous_hypothesis] = frame_id
                record_detection(gt_codes[gt_index], distance, "MATCH")

            distances[gt_masked, :] = np.nan
            distances[:, tracker_masked] = np.nan
            rows, columns = linear_sum_assignment(costs_for_matching)
            for gt_index, tracker_index in zip(rows, columns):
                distance = distances[gt_index, tracker_index]
                if not np.isfinite(distance):
                    continue

                object_id = gt_ids[gt_index]
                hypothesis_id = tracker_ids[tracker_index]
                is_switch = (
                    object_id in object_to_hypothesis
                    and object_to_hypothesis[object_id] != hypothesis_id
                    and object_id in last_occurrence
                )
                event_type = "SWITCH" if is_switch else "MATCH"
                if is_switch and hypothesis_id not in hypothesis_history:
                    type_counts["ASCEND"] += 1

                is_transfer = (
                    hypothesis_id in hypothesis_to_object
                    and hypothesis_to_object[hypothesis_id] != object_id
                )
                if is_transfer:
                    if object_id not in last_match:
                        type_counts["MIGRATE"] += 1
                    type_counts["TRANSFER"] += 1

                hypothesis_history[hypothesis_id] = frame_id
                last_match[object_id] = frame_id
                gt_masked[gt_index] = True
                tracker_masked[tracker_index] = True
                object_to_hypothesis[object_id] = hypothesis_id
                hypothesis_to_object[hypothesis_id] = object_id
                record_detection(gt_codes[gt_index], distance, event_type)

        missed_codes = gt_codes[~gt_masked]
        type_counts["MISS"] += len(missed_codes)
        missed_after_track[missed_codes[was_tracked[missed_codes]]] = True
        type_counts["FP"] += int((~tracker_masked).sum())
        last_occurrence.update((object_id, frame_id) for object_id in gt_ids)
        last_update_frame_id = frame_id

    return type_counts, tracked_counts, num_fragmentations, distance_sum


def compute_clear_identity(prepared, distance_threshold=0.5):
    """Compute compact CLEAR and Identity statistics from prepared arrays."""
    type_counts, tracked_counts, num_fragmentations, distance_sum = (
        _compute_clear_statistics(prepared, distance_threshold)
    )
    num_detections = type_counts["MATCH"] + type_counts["SWITCH"]
    track_ratios_array = math_util.quiet_divide(
        tracked_counts,
        prepared.gt_counts,
    )
    track_ratios = pd.Series(track_ratios_array, index=prepared.unique_gt_ids)

    identity_assignment = _identity_assignment(prepared, distance_threshold)
    assignment_rows = identity_assignment["rids"]
    assignment_columns = identity_assignment["cids"]
    idfp = identity_assignment["fpmatrix"][assignment_rows, assignment_columns].sum()
    idfn = identity_assignment["fnmatrix"][assignment_rows, assignment_columns].sum()
    idtp = prepared.num_gt_detections - idfn
    motp = math_util.quiet_divide(
        distance_sum,
        num_detections,
    )
    stats = {
        "num_frames": len(prepared.frame_ids),
        "obj_frequencies": pd.Series(prepared.gt_counts, index=prepared.unique_gt_ids),
        "pred_frequencies": pd.Series(
            prepared.tracker_counts,
            index=prepared.unique_tracker_ids,
        ),
        "num_matches": type_counts["MATCH"],
        "num_switches": type_counts["SWITCH"],
        "num_transfer": type_counts["TRANSFER"],
        "num_ascend": type_counts["ASCEND"],
        "num_migrate": type_counts["MIGRATE"],
        "num_false_positives": type_counts["FP"],
        "num_misses": type_counts["MISS"],
        "num_detections": num_detections,
        "num_objects": prepared.num_gt_detections,
        "num_predictions": prepared.num_tracker_detections,
        "num_gt_ids": len(prepared.unique_gt_ids),
        "num_dt_ids": len(prepared.unique_tracker_ids),
        "num_unique_objects": len(prepared.unique_gt_ids),
        "track_ratios": track_ratios,
        "mostly_tracked": int(np.count_nonzero(track_ratios_array >= 0.8)),
        "partially_tracked": int(
            np.count_nonzero((track_ratios_array >= 0.2) & (track_ratios_array < 0.8))
        ),
        "mostly_lost": int(np.count_nonzero(track_ratios_array < 0.2)),
        "num_fragmentations": num_fragmentations,
        "motp": motp,
        "id_global_assignment": identity_assignment,
        "idfp": idfp,
        "idfn": idfn,
        "idtp": idtp,
    }
    stats.update(
        {
            "mota": 1.0
            - math_util.quiet_divide(
                stats["num_misses"]
                + stats["num_switches"]
                + stats["num_false_positives"],
                stats["num_objects"],
            ),
            "precision": math_util.quiet_divide(
                num_detections,
                num_detections + stats["num_false_positives"],
            ),
            "recall": math_util.quiet_divide(
                num_detections,
                stats["num_objects"],
            ),
            "idp": math_util.quiet_divide(idtp, idtp + idfp),
            "idr": math_util.quiet_divide(idtp, idtp + idfn),
            "idf1": math_util.quiet_divide(
                2 * idtp,
                stats["num_objects"] + stats["num_predictions"],
            ),
        }
    )
    accumulator = MOTAccumulator()
    deferred_frames = list(
        zip(
            prepared.gt_ids,
            prepared.tracker_ids,
            prepared.frame_ids,
            prepared.similarities,
        )
    )
    accumulator._defer_clear_event_updates(
        deferred_frames,
        distance_threshold,
        stats,
    )
    return ClearIdentityResult(stats=stats, accumulator=accumulator)


def evaluate_sequence(
    prepared,
    clear_threshold=0.5,
    hota_thresholds=DEFAULT_HOTA_THRESHOLDS,
):
    """Compute CLEAR, Identity, and HOTA from one prepared sequence."""
    return SequenceResult(
        prepared=prepared,
        clear_identity=compute_clear_identity(prepared, clear_threshold),
        hota=compute_hota(prepared, hota_thresholds),
    )


def combine_clear_identity(results):
    """Combine compact CLEAR and Identity sequence results."""
    stats = [result.stats for result in results]
    additive = (
        "num_frames",
        "num_matches",
        "num_switches",
        "num_transfer",
        "num_ascend",
        "num_migrate",
        "num_false_positives",
        "num_misses",
        "num_detections",
        "num_objects",
        "num_predictions",
        "num_gt_ids",
        "num_dt_ids",
        "num_unique_objects",
        "mostly_tracked",
        "partially_tracked",
        "mostly_lost",
        "num_fragmentations",
        "idfp",
        "idfn",
        "idtp",
    )
    combined = {name: sum(sequence[name] for sequence in stats) for name in additive}
    combined.update(
        {
            "motp": math_util.quiet_divide(
                sum(sequence["motp"] * sequence["num_detections"] for sequence in stats),
                combined["num_detections"],
            ),
            "mota": 1.0
            - math_util.quiet_divide(
                combined["num_misses"]
                + combined["num_switches"]
                + combined["num_false_positives"],
                combined["num_objects"],
            ),
            "precision": math_util.quiet_divide(
                combined["num_detections"],
                combined["num_detections"] + combined["num_false_positives"],
            ),
            "recall": math_util.quiet_divide(
                combined["num_detections"],
                combined["num_objects"],
            ),
            "idp": math_util.quiet_divide(
                combined["idtp"],
                combined["idtp"] + combined["idfp"],
            ),
            "idr": math_util.quiet_divide(
                combined["idtp"],
                combined["idtp"] + combined["idfn"],
            ),
            "idf1": math_util.quiet_divide(
                2 * combined["idtp"],
                combined["num_objects"] + combined["num_predictions"],
            ),
        }
    )
    return combined


def _resolve_workers(n_jobs, num_sequences):
    if n_jobs is None:
        return min(num_sequences, max(1, (os.cpu_count() or 1) - 2))
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")
    return n_jobs


def prepare_many(sequences, n_jobs=None):
    """Prepare named ``(ground_truth, tracker)`` pairs in parallel."""
    names = list(sequences)
    workers = _resolve_workers(n_jobs, len(names))

    def prepare(name):
        ground_truth, tracker = sequences[name]
        return prepare_sequence(ground_truth, tracker)

    if workers == 1 or len(names) < 2:
        prepared = [prepare(name) for name in names]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            prepared = list(executor.map(prepare, names))
    return OrderedDict(zip(names, prepared))


def evaluate_many(
    sequences,
    clear_threshold=0.5,
    hota_thresholds=DEFAULT_HOTA_THRESHOLDS,
    n_jobs=None,
):
    """Evaluate complete sequences with one worker task per sequence."""
    names = list(sequences)
    workers = _resolve_workers(n_jobs, len(names))

    def evaluate(name):
        sequence = sequences[name]
        prepared = (
            sequence
            if isinstance(sequence, PreparedSequence)
            else prepare_sequence(*sequence)
        )
        return evaluate_sequence(prepared, clear_threshold, hota_thresholds)

    if workers == 1 or len(names) < 2:
        results = [evaluate(name) for name in names]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(evaluate, names))
    sequence_results = OrderedDict(zip(names, results))
    clear_results = [result.clear_identity for result in results]
    hota_results = [result.hota for result in results]
    return BatchResult(
        sequences=sequence_results,
        overall_clear_identity=combine_clear_identity(clear_results),
        overall_hota=HOTAResult.combine(hota_results),
    )
