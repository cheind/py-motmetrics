from pathlib import Path

import numpy as np
import pytest

import motmetrics as mm

trackeval = pytest.importorskip(
    "trackeval",
    reason="TrackEval is installed by the dedicated parity CI job",
)


DATA_DIR = Path(__file__).parents[1] / "data"
SEQUENCE_NAMES = ("TUD-Campus", "TUD-Stadtmitte")
HOTA_ALPHAS = np.arange(0.05, 0.99, 0.05)
BOX_COLUMNS = ["X", "Y", "Width", "Height"]

CLEAR_FIELD_MAP = {
    "MOTA": "mota",
    "CLR_Re": "recall",
    "CLR_Pr": "precision",
    "CLR_TP": "num_detections",
    "CLR_FN": "num_misses",
    "CLR_FP": "num_false_positives",
    "IDSW": "num_switches",
    "MT": "mostly_tracked",
    "PT": "partially_tracked",
    "ML": "mostly_lost",
    "Frag": "num_fragmentations",
}
IDENTITY_FIELD_MAP = {
    "IDF1": "idf1",
    "IDR": "idr",
    "IDP": "idp",
    "IDTP": "idtp",
    "IDFN": "idfn",
    "IDFP": "idfp",
}


def _load_sequence(sequence_name):
    sequence_dir = DATA_DIR / sequence_name
    return (
        mm.io.loadtxt(sequence_dir / "gt.txt"),
        mm.io.loadtxt(sequence_dir / "test.txt"),
    )


def _frame_data(dataframe, frame_id, id_map):
    try:
        frame = dataframe.xs(frame_id, level="FrameId")
    except KeyError:
        return np.empty(0, dtype=int), np.empty((0, 4), dtype=float)

    ids = np.asarray([id_map[value] for value in frame.index], dtype=int)
    boxes = frame[BOX_COLUMNS].to_numpy(dtype=float)
    return ids, boxes


def _to_trackeval_data(ground_truth, tracker):
    ground_truth_ids = {
        value: index
        for index, value in enumerate(sorted(ground_truth.index.get_level_values("Id").unique()))
    }
    tracker_ids = {
        value: index
        for index, value in enumerate(sorted(tracker.index.get_level_values("Id").unique()))
    }
    frame_ids = sorted(
        set(ground_truth.index.get_level_values("FrameId"))
        | set(tracker.index.get_level_values("FrameId"))
    )

    gt_ids = []
    tracker_ids_by_frame = []
    similarity_scores = []
    for frame_id in frame_ids:
        gt_ids_t, gt_boxes_t = _frame_data(ground_truth, frame_id, ground_truth_ids)
        tracker_ids_t, tracker_boxes_t = _frame_data(tracker, frame_id, tracker_ids)
        gt_ids.append(gt_ids_t)
        tracker_ids_by_frame.append(tracker_ids_t)
        similarity_scores.append(
            mm.distances.iou_matrix(gt_boxes_t, tracker_boxes_t, return_dist=False)
        )

    return {
        "num_timesteps": len(frame_ids),
        "num_gt_ids": len(ground_truth_ids),
        "num_tracker_ids": len(tracker_ids),
        "num_gt_dets": sum(len(ids) for ids in gt_ids),
        "num_tracker_dets": sum(len(ids) for ids in tracker_ids_by_frame),
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids_by_frame,
        "similarity_scores": similarity_scores,
    }


def _compute_py_motmetrics(sequences):
    metric_host = mm.metrics.create()
    names = list(sequences)
    clear_accumulators = [
        mm.utils.compare_to_groundtruth(ground_truth, tracker, "iou", distth=0.5)
        for ground_truth, tracker in sequences.values()
    ]
    clear_and_identity = metric_host.compute_many(
        clear_accumulators,
        metrics=[
            *CLEAR_FIELD_MAP.values(),
            "motp",
            *IDENTITY_FIELD_MAP.values(),
        ],
        names=names,
        generate_overall=True,
    )

    hota_by_alpha = []
    hota_accumulators = {
        name: mm.utils.compare_to_groundtruth_reweighting(
            ground_truth,
            tracker,
            "iou",
            distth=HOTA_ALPHAS,
        )
        for name, (ground_truth, tracker) in sequences.items()
    }
    for alpha_index in range(len(HOTA_ALPHAS)):
        hota_by_alpha.append(
            metric_host.compute_many(
                [hota_accumulators[name][alpha_index] for name in names],
                metrics=["hota_alpha", "deta_alpha", "assa_alpha"],
                names=names,
                generate_overall=True,
            )
        )

    results = {}
    for name in [*names, "OVERALL"]:
        result = {
            trackeval_name: clear_and_identity.loc[name, py_motmetrics_name]
            for trackeval_name, py_motmetrics_name in CLEAR_FIELD_MAP.items()
        }
        result["MOTP"] = 1.0 - clear_and_identity.loc[name, "motp"]
        result.update(
            {
                trackeval_name: clear_and_identity.loc[name, py_motmetrics_name]
                for trackeval_name, py_motmetrics_name in IDENTITY_FIELD_MAP.items()
            }
        )
        result.update(
            {
                "HOTA": np.asarray([summary.loc[name, "hota_alpha"] for summary in hota_by_alpha]),
                "DetA": np.asarray([summary.loc[name, "deta_alpha"] for summary in hota_by_alpha]),
                "AssA": np.asarray([summary.loc[name, "assa_alpha"] for summary in hota_by_alpha]),
            }
        )
        results[name] = result
    return results


def _compute_trackeval(sequences):
    metrics = {
        "HOTA": trackeval.metrics.HOTA(),
        "CLEAR": trackeval.metrics.CLEAR({"THRESHOLD": 0.5}),
        "Identity": trackeval.metrics.Identity({"THRESHOLD": 0.5}),
    }
    sequence_results = {
        name: {
            metric_name: metric.eval_sequence(_to_trackeval_data(ground_truth, tracker))
            for metric_name, metric in metrics.items()
        }
        for name, (ground_truth, tracker) in sequences.items()
    }

    results = {}
    for name, family_results in sequence_results.items():
        results[name] = {
            **family_results["HOTA"],
            **family_results["CLEAR"],
            **family_results["Identity"],
        }
    results["OVERALL"] = {}
    for metric_name, metric in metrics.items():
        combined = metric.combine_sequences(
            {name: sequence_results[name][metric_name] for name in sequences}
        )
        results["OVERALL"].update(combined)
    return results


def test_metrics_match_trackeval_on_bundled_tud_sequences():
    sequences = {name: _load_sequence(name) for name in SEQUENCE_NAMES}
    py_motmetrics_results = _compute_py_motmetrics(sequences)
    trackeval_results = _compute_trackeval(sequences)

    fields = [
        "HOTA",
        "DetA",
        "AssA",
        "MOTA",
        "MOTP",
        "CLR_Re",
        "CLR_Pr",
        "CLR_TP",
        "CLR_FN",
        "CLR_FP",
        "IDSW",
        "MT",
        "PT",
        "ML",
        "Frag",
        "IDF1",
        "IDR",
        "IDP",
        "IDTP",
        "IDFN",
        "IDFP",
    ]
    for sequence_name in [*SEQUENCE_NAMES, "OVERALL"]:
        for field in fields:
            np.testing.assert_allclose(
                py_motmetrics_results[sequence_name][field],
                trackeval_results[sequence_name][field],
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"{sequence_name}: {field}",
            )
