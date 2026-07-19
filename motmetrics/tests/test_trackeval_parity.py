import os
from pathlib import Path

import numpy as np
import pytest

import motmetrics as mm
import motmetrics._distances as distances
import motmetrics._evaluation as evaluation
import motmetrics._io as io
import motmetrics._metrics as metrics

trackeval = pytest.importorskip(
    "trackeval",
    reason="TrackEval is installed by the dedicated parity CI job",
)


DATA_DIR = Path(__file__).parents[1] / "data"
SEQUENCE_NAMES = ("TUD-Campus", "TUD-Stadtmitte")
HOTA_ALPHAS = np.arange(0.05, 0.99, 0.05)
BOX_COLUMNS = ["X", "Y", "Width", "Height"]
PARITY_TOLERANCE = 1e-6

CLEAR_FIELD_MAP = {
    "MOTA": "mota",
    "MODA": "moda",
    "CLR_Re": "recall",
    "CLR_Pr": "precision",
    "MTR": "mtr",
    "PTR": "ptr",
    "MLR": "mlr",
    "sMOTA": "smota",
    "CLR_F1": "clr_f1",
    "FP_per_frame": "fp_per_frame",
    "CLR_TP": "num_detections",
    "CLR_FN": "num_misses",
    "CLR_FP": "num_false_positives",
    "IDSW": "num_switches",
    "MT": "mostly_tracked",
    "PT": "partially_tracked",
    "ML": "mostly_lost",
    "Frag": "num_fragmentations",
}
HOTA_FIELD_MAP = {
    "HOTA": "hota_alpha",
    "DetA": "deta_alpha",
    "AssA": "assa_alpha",
    "DetRe": "detre_alpha",
    "DetPr": "detpr_alpha",
    "AssRe": "assre_alpha",
    "AssPr": "asspr_alpha",
    "LocA": "loca_alpha",
    "OWTA": "owta_alpha",
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
        io.loadtxt(sequence_dir / "gt.txt"),
        io.loadtxt(sequence_dir / "test.txt"),
    )


def _frame_data(data, frame_id, id_map):
    mask = data.frame_ids == frame_id
    if not np.any(mask):
        return np.empty(0, dtype=int), np.empty((0, 4), dtype=float)

    ids = np.asarray([id_map[value] for value in data.ids[mask]], dtype=int)
    boxes = data.values(BOX_COLUMNS)[mask]
    return ids, boxes


def _to_trackeval_data(ground_truth, tracker):
    ground_truth_ids = {
        value: index
        for index, value in enumerate(np.unique(ground_truth.ids))
    }
    tracker_ids = {
        value: index
        for index, value in enumerate(np.unique(tracker.ids))
    }
    frame_ids = np.union1d(ground_truth.frame_ids, tracker.frame_ids)

    gt_ids = []
    tracker_ids_by_frame = []
    similarity_scores = []
    for frame_id in frame_ids:
        gt_ids_t, gt_boxes_t = _frame_data(ground_truth, frame_id, ground_truth_ids)
        tracker_ids_t, tracker_boxes_t = _frame_data(tracker, frame_id, tracker_ids)
        gt_ids.append(gt_ids_t)
        tracker_ids_by_frame.append(tracker_ids_t)
        similarity_scores.append(
            distances.iou_matrix(gt_boxes_t, tracker_boxes_t, return_dist=False)
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
    names = list(sequences)
    requested_metrics = [
        *CLEAR_FIELD_MAP.values(),
        "motp",
        *IDENTITY_FIELD_MAP.values(),
    ]
    metric_partials = {}
    hota_summaries = {}
    for name, (ground_truth, tracker) in sequences.items():
        prepared = evaluation._prepare_iou_sequence_data(ground_truth, tracker, 0.5)
        metric_partials[name] = metrics._compute_metrics(
            prepared.accumulator,
            metric_names=requested_metrics,
        )
        hota_summaries[name] = evaluation._compute_prepared_hota_sequence_summary(
            prepared,
            HOTA_ALPHAS,
        )
    metric_partials["OVERALL"] = metrics._compute_overall(
        list(metric_partials.values()),
        metric_names=requested_metrics,
    )
    hota_summaries["OVERALL"] = evaluation._combine_hota_sequence_summaries(
        hota_summaries.values()
    )

    results = {}
    for name in [*names, "OVERALL"]:
        result = {
            trackeval_name: metric_partials[name][py_motmetrics_name]
            for trackeval_name, py_motmetrics_name in CLEAR_FIELD_MAP.items()
        }
        result["MOTP"] = 1.0 - metric_partials[name]["motp"]
        result.update(
            {
                trackeval_name: metric_partials[name][py_motmetrics_name]
                for trackeval_name, py_motmetrics_name in IDENTITY_FIELD_MAP.items()
            }
        )
        result.update(
            {
                trackeval_name: hota_summaries[name][py_motmetrics_name]
                for trackeval_name, py_motmetrics_name in HOTA_FIELD_MAP.items()
            }
        )
        results[name] = result
    return results


def _compute_trackeval(sequences):
    metrics = {
        "HOTA": trackeval.metrics.HOTA(),
        "CLEAR": trackeval.metrics.CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
        "Identity": trackeval.metrics.Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
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


def _metric_comparison(py_motmetrics_value, trackeval_value):
    py_motmetrics_array = np.asarray(py_motmetrics_value, dtype=float)
    trackeval_array = np.asarray(trackeval_value, dtype=float)
    py_motmetrics_summary = float(np.mean(py_motmetrics_array))
    trackeval_summary = float(np.mean(trackeval_array))

    if py_motmetrics_array.shape != trackeval_array.shape:
        return py_motmetrics_summary, trackeval_summary, float("inf")

    finite_pairs = np.isfinite(py_motmetrics_array) & np.isfinite(trackeval_array)
    same_nonfinite = (
        (np.isnan(py_motmetrics_array) & np.isnan(trackeval_array))
        | (np.isinf(py_motmetrics_array) & (py_motmetrics_array == trackeval_array))
    )
    differences = np.full(py_motmetrics_array.shape, np.inf, dtype=float)
    differences[finite_pairs] = np.abs(
        py_motmetrics_array[finite_pairs] - trackeval_array[finite_pairs]
    )
    differences[same_nonfinite] = 0.0
    max_difference = float(np.max(differences)) if differences.size else 0.0
    return py_motmetrics_summary, trackeval_summary, max_difference


def _render_comparison_table(rows):
    headers = ("Dataset", "Metric", "py-motmetrics", "TrackEval", "max abs diff", "Status")
    rendered_rows = [
        (
            sequence_name,
            field,
            f"{py_motmetrics_value:.12g}",
            f"{trackeval_value:.12g}",
            f"{max_difference:.3e}",
            "PASS" if max_difference <= PARITY_TOLERANCE else "FAIL",
        )
        for sequence_name, field, py_motmetrics_value, trackeval_value, max_difference in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rendered_rows))
        for index in range(len(headers))
    ]

    def render_row(row):
        return " | ".join(value.ljust(width) for value, width in zip(row, widths))

    separator = "-+-".join("-" * width for width in widths)
    table = [render_row(headers), separator, *(render_row(row) for row in rendered_rows)]
    return "\n".join(
        [
            f"TrackEval parity (absolute tolerance: {PARITY_TOLERANCE:.0e})",
            "HOTA-family values are alpha means; their difference is the maximum over all alphas.",
            *table,
        ]
    )


def _write_github_summary(rows):
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    lines = [
        "## TrackEval parity",
        "",
        f"Maximum permitted absolute difference: `{PARITY_TOLERANCE:.0e}`.",
        "HOTA-family values are alpha means; their difference is the maximum over all alphas.",
        "",
        "| Dataset | Metric | py-motmetrics | TrackEval | max abs diff | Status |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for sequence_name, field, py_motmetrics_value, trackeval_value, max_difference in rows:
        status = "PASS" if max_difference <= PARITY_TOLERANCE else "FAIL"
        lines.append(
            f"| {sequence_name} | {field} | {py_motmetrics_value:.12g} | "
            f"{trackeval_value:.12g} | {max_difference:.3e} | {status} |"
        )
    with Path(summary_path).open("a", encoding="utf-8") as summary_file:
        summary_file.write("\n".join(lines) + "\n")


def test_metrics_match_trackeval_on_bundled_tud_sequences():
    sequences = {name: _load_sequence(name) for name in SEQUENCE_NAMES}
    py_motmetrics_results = _compute_py_motmetrics(sequences)
    trackeval_results = _compute_trackeval(sequences)

    fields = [
        "HOTA",
        "DetA",
        "AssA",
        "DetRe",
        "DetPr",
        "AssRe",
        "AssPr",
        "LocA",
        "OWTA",
        "MOTA",
        "MOTP",
        "MODA",
        "CLR_Re",
        "CLR_Pr",
        "MTR",
        "PTR",
        "MLR",
        "sMOTA",
        "CLR_F1",
        "FP_per_frame",
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
    rows = []
    for sequence_name in [*SEQUENCE_NAMES, "OVERALL"]:
        for field in fields:
            py_motmetrics_value, trackeval_value, max_difference = _metric_comparison(
                py_motmetrics_results[sequence_name][field],
                trackeval_results[sequence_name][field],
            )
            rows.append(
                (sequence_name, field, py_motmetrics_value, trackeval_value, max_difference)
            )

    print("\n" + _render_comparison_table(rows))
    _write_github_summary(rows)

    failures = [
        f"{sequence_name}: {field} ({max_difference:.3e})"
        for sequence_name, field, _, _, max_difference in rows
        if max_difference > PARITY_TOLERANCE
    ]
    if failures:
        pytest.fail(
            f"TrackEval parity exceeded {PARITY_TOLERANCE:.0e}:\n" + "\n".join(failures),
            pytrace=False,
        )

    public_summary = mm.evaluate_motchallenge(DATA_DIR, DATA_DIR, progress=False)
    public_rows = dict(zip(public_summary.index, public_summary._rows))
    hota_public_names = {
        trackeval_name: evaluation.HOTA_SUMMARY_METRICS[alpha_name]
        for trackeval_name, alpha_name in HOTA_FIELD_MAP.items()
    }
    clear_public_names = {
        trackeval_name: py_motmetrics_name
        for trackeval_name, py_motmetrics_name in CLEAR_FIELD_MAP.items()
        if py_motmetrics_name in public_summary.columns
    }
    for sequence_name in [*SEQUENCE_NAMES, "OVERALL"]:
        for trackeval_name, public_name in hota_public_names.items():
            assert public_rows[sequence_name][public_name] == pytest.approx(
                np.mean(trackeval_results[sequence_name][trackeval_name]),
                abs=PARITY_TOLERANCE,
            )
        for trackeval_name, public_name in clear_public_names.items():
            assert public_rows[sequence_name][public_name] == pytest.approx(
                trackeval_results[sequence_name][trackeval_name],
                abs=PARITY_TOLERANCE,
            )


def test_public_evaluator_matches_full_trackeval_mot17_protocol(tmp_path):
    sequence_name = "MOT17-SYNTH"
    tracker_name = "synthetic-tracker"
    ground_truth_root = tmp_path / "ground-truth"
    ground_truth_file = ground_truth_root / sequence_name / "gt" / "gt.txt"
    tracker_root = tmp_path / "trackers"
    tracker_data = tracker_root / tracker_name / "data"
    tracker_file = tracker_data / "{}.txt".format(sequence_name)
    ground_truth_file.parent.mkdir(parents=True)
    tracker_data.mkdir(parents=True)
    ground_truth_file.write_text(
        "\n".join((
            "1,1,1,1,10,10,1,1,1",
            "1,90,101,1,10,10,0,8,1",
            "2,1,2,1,10,10,1,1,1",
            "2,91,201,1,10,10,0,7,1",
            "2,3,301,1,10,10,0,1,1",
            "3,2,401,1,10,10,1,1,1",
            "3,92,501,1,10,10,0,2,1",
            "3,93,601,1,10,10,0,12,1",
            "3,94,701,1,10,10,0,3,1",
            "4,2,402,1,10,10,1,1,1",
            "5,2,403,1,10,10,1,1,1",
        )),
        encoding="utf-8",
    )
    tracker_file.write_text(
        "\n".join((
            "1,10,1,1,10,10,1,1,1",
            "1,20,101,1,10,10,1,1,1",
            "1,90,801,1,10,10,1,1,1",
            "2,10,2,1,10,10,1,1,1",
            "2,20,201,1,10,10,1,1,1",
            "2,30,301,1,10,10,1,1,1",
            "3,40,401,1,10,10,1,1,1",
            "3,50,501,1,10,10,1,1,1",
            "3,60,601,1,10,10,1,1,1",
            "3,70,701,1,10,10,1,1,1",
            "5,40,403,1,10,10,1,1,1",
        )),
        encoding="utf-8",
    )

    dataset = trackeval.datasets.MotChallenge2DBox({
        "GT_FOLDER": str(ground_truth_root),
        "TRACKERS_FOLDER": str(tracker_root),
        "OUTPUT_FOLDER": str(tmp_path / "output"),
        "TRACKERS_TO_EVAL": [tracker_name],
        "TRACKER_SUB_FOLDER": "data",
        "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": "MOT17",
        "SPLIT_TO_EVAL": "train",
        "DO_PREPROC": True,
        "SEQ_INFO": {sequence_name: 5},
        "SKIP_SPLIT_FOL": True,
        "PRINT_CONFIG": False,
    })
    preprocessed = dataset.get_preprocessed_seq_data(
        dataset.get_raw_seq_data(tracker_name, sequence_name),
        "pedestrian",
    )
    metric_objects = {
        "HOTA": trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
        "CLEAR": trackeval.metrics.CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
        "Identity": trackeval.metrics.Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
        "Count": trackeval.metrics.Count({"PRINT_CONFIG": False}),
    }
    trackeval_sequence = {
        family: metric.eval_sequence(preprocessed)
        for family, metric in metric_objects.items()
    }
    trackeval_overall = {
        family: metric.combine_sequences({sequence_name: trackeval_sequence[family]})
        for family, metric in metric_objects.items()
    }

    summary = mm.evaluate_motchallenge(
        ground_truth_root,
        tracker_data,
        progress=False,
    )
    for row_name, expected in (
        (sequence_name, trackeval_sequence),
        ("OVERALL", trackeval_overall),
    ):
        expected_values = _public_trackeval_values(expected)
        assert set(expected_values) == set(summary.columns) - {
            "num_transfer",
            "num_ascend",
            "num_migrate",
        }
        for metric_name, expected_value in expected_values.items():
            assert summary[row_name, metric_name] == pytest.approx(
                expected_value,
                abs=PARITY_TOLERANCE,
            )


def _public_trackeval_values(results):
    clear = results["CLEAR"]
    identity = results["Identity"]
    hota = results["HOTA"]
    count = results["Count"]
    return {
        "idf1": identity["IDF1"],
        "idp": identity["IDP"],
        "idr": identity["IDR"],
        "recall": clear["CLR_Re"],
        "precision": clear["CLR_Pr"],
        "num_unique_objects": count["GT_IDs"],
        "mostly_tracked": clear["MT"],
        "partially_tracked": clear["PT"],
        "mostly_lost": clear["ML"],
        "mtr": clear["MTR"],
        "ptr": clear["PTR"],
        "mlr": clear["MLR"],
        "num_false_positives": clear["CLR_FP"],
        "num_misses": clear["CLR_FN"],
        "num_switches": clear["IDSW"],
        "num_fragmentations": clear["Frag"],
        "mota": clear["MOTA"],
        "moda": clear["MODA"],
        "motp": 1 - clear["MOTP"],
        "smota": clear["sMOTA"],
        "clr_f1": clear["CLR_F1"],
        "fp_per_frame": clear["FP_per_frame"],
        "hota": np.mean(hota["HOTA"]),
        "deta": np.mean(hota["DetA"]),
        "assa": np.mean(hota["AssA"]),
        "detre": np.mean(hota["DetRe"]),
        "detpr": np.mean(hota["DetPr"]),
        "assre": np.mean(hota["AssRe"]),
        "asspr": np.mean(hota["AssPr"]),
        "loca": np.mean(hota["LocA"]),
        "owta": np.mean(hota["OWTA"]),
    }
