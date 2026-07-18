"""Compute confidence-ranked detection AP with custom matching semantics.

This example does not use the built-in CLEAR or HOTA assignment. Predictions
are ranked globally by confidence and greedily matched to the best unmatched
ground-truth box in the same frame. The family also creates and combines its
own precision-recall state, which the fast MOTChallenge path does not produce.

Run this example with MOTChallenge files or evaluation roots whose prediction
rows contain meaningful confidence values:

    python examples/custom_metric_detection_average_precision.py path/to/gt path/to/predictions
"""

import argparse
from typing import NamedTuple

import numpy as np

import motmetrics as mm


class APPartial(NamedTuple):
    """Picklable state required to recompute AP across sequences."""

    scores: np.ndarray
    true_positives: np.ndarray
    ground_truth_count: int


class ConfidenceRankedDetectionAP(mm.MetricFamily):
    """Detection AP using confidence-ordered greedy matching at one IoU."""

    name = "confidence_ranked_detection_ap"
    metric_names = ("det_ap",)
    requirements = frozenset()
    display_names = {"det_ap": "DetAP"}
    formatters = {"det_ap": "{:.1%}".format}

    def __init__(self, iou_threshold=0.5):
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0")
        self.iou_threshold = float(iou_threshold)

    def evaluate_sequence(self, sequence, intermediates):
        del intermediates
        ground_truth_by_frame = _group_ground_truth(sequence.ground_truth)
        tracker = sequence.tracker
        scores = np.asarray(tracker.confidence, dtype=float)
        if np.any(~np.isfinite(scores)):
            raise ValueError("Detection AP requires finite tracker confidences")

        ranked_indices = np.argsort(-scores, kind="stable")
        ranked_scores = scores[ranked_indices].copy()
        true_positives = np.zeros(len(ranked_indices), dtype=np.bool_)

        for rank, detection_index in enumerate(ranked_indices):
            frame_id = int(tracker.frame_ids[detection_index])
            frame_data = ground_truth_by_frame.get(frame_id)
            if frame_data is None:
                continue

            ground_truth_boxes, matched = frame_data
            available = np.flatnonzero(~matched)
            if len(available) == 0:
                continue

            similarities = _box_iou(
                tracker.boxes[detection_index],
                ground_truth_boxes[available],
            )
            best_available_index = int(np.argmax(similarities))
            if similarities[best_available_index] >= self.iou_threshold:
                matched[available[best_available_index]] = True
                true_positives[rank] = True

        return APPartial(
            scores=ranked_scores,
            true_positives=true_positives,
            ground_truth_count=len(sequence.ground_truth),
        )

    def summarize(self, partial):
        det_ap = _average_precision(partial)
        if not 0.0 <= det_ap <= 1.0:
            raise ValueError(
                "Detection AP must be between 0.0 and 1.0, got {!r}".format(
                    det_ap
                )
            )
        return {"det_ap": det_ap}

    def combine(self, partials):
        return self.summarize(APPartial(
            scores=_concatenate(partials, "scores", dtype=float),
            true_positives=_concatenate(
                partials,
                "true_positives",
                dtype=np.bool_,
            ),
            ground_truth_count=sum(
                partial.ground_truth_count
                for partial in partials
            ),
        ))


def _group_ground_truth(ground_truth):
    grouped_boxes = {}
    for frame_id, box in zip(ground_truth.frame_ids, ground_truth.boxes):
        grouped_boxes.setdefault(int(frame_id), []).append(box)
    return {
        frame_id: (
            np.asarray(boxes, dtype=float),
            np.zeros(len(boxes), dtype=np.bool_),
        )
        for frame_id, boxes in grouped_boxes.items()
    }


def _box_iou(box, boxes):
    intersection_left = np.maximum(box[0], boxes[:, 0])
    intersection_top = np.maximum(box[1], boxes[:, 1])
    intersection_right = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
    intersection_bottom = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])
    intersection = (
        np.maximum(0.0, intersection_right - intersection_left)
        * np.maximum(0.0, intersection_bottom - intersection_top)
    )
    box_area = max(0.0, box[2]) * max(0.0, box[3])
    boxes_area = np.maximum(0.0, boxes[:, 2]) * np.maximum(0.0, boxes[:, 3])
    union = box_area + boxes_area - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros(len(boxes), dtype=float),
        where=union > 0,
    )


def _average_precision(partial):
    if partial.ground_truth_count == 0 or len(partial.scores) == 0:
        return 0.0

    order = np.argsort(-partial.scores, kind="stable")
    true_positives = np.cumsum(partial.true_positives[order], dtype=float)
    false_positives = np.cumsum(~partial.true_positives[order], dtype=float)
    recall = true_positives / partial.ground_truth_count
    precision = true_positives / np.maximum(1.0, true_positives + false_positives)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changed_recall = recall[1:] != recall[:-1]
    return float(np.sum(
        (recall[1:] - recall[:-1])[changed_recall]
        * precision[1:][changed_recall]
    ))


def _concatenate(partials, attribute, dtype):
    arrays = [getattr(partial, attribute) for partial in partials]
    if not arrays:
        return np.empty(0, dtype=dtype)
    return np.concatenate(arrays).astype(dtype, copy=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", help="Ground-truth file or evaluation root")
    parser.add_argument("predictions", help="Prediction file or evaluation root")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Greedy matching IoU threshold (default: 0.5)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Sequence worker processes (default: 1)",
    )
    args = parser.parse_args()

    summary = mm.evaluate_motchallenge(
        args.ground_truth,
        args.predictions,
        n_jobs=args.jobs,
        extra_metric_families=ConfidenceRankedDetectionAP(
            iou_threshold=args.iou_threshold,
        ),
        progress=False,
    )
    print(summary)


if __name__ == "__main__":
    main()
