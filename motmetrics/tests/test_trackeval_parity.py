import os
import statistics
import time
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
PARITY_TOLERANCE = 1e-6
TIMING_REPEATS = 3

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
        "CLEAR": trackeval.metrics.CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
        "Identity": trackeval.metrics.Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
    }
    sequence_results = {}
    for name, (ground_truth, tracker) in sequences.items():
        sequence_data = _to_trackeval_data(ground_truth, tracker)
        sequence_results[name] = {
            metric_name: metric.eval_sequence(sequence_data)
            for metric_name, metric in metrics.items()
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


def _measure_execution_times(sequences):
    evaluators = {
        "py-motmetrics": _compute_py_motmetrics,
        "TrackEval": _compute_trackeval,
    }

    # Warm both implementations before measuring so imports, solver setup, and
    # allocator initialization do not dominate these small bundled sequences.
    for evaluator in evaluators.values():
        evaluator(sequences)

    results = {}
    timings = {name: [] for name in evaluators}
    evaluator_names = list(evaluators)
    for repeat_index in range(TIMING_REPEATS):
        # Alternate order to reduce systematic first/second-run bias.
        run_order = evaluator_names if repeat_index % 2 == 0 else list(reversed(evaluator_names))
        for name in run_order:
            start_time = time.perf_counter()
            results[name] = evaluators[name](sequences)
            timings[name].append(time.perf_counter() - start_time)

    return results["py-motmetrics"], results["TrackEval"], timings


def _relative_time_summary(timings):
    py_motmetrics_median = statistics.median(timings["py-motmetrics"])
    trackeval_median = statistics.median(timings["TrackEval"])
    if py_motmetrics_median <= trackeval_median:
        factor = trackeval_median / py_motmetrics_median
        return f"py-motmetrics was {factor:.2f}x faster than TrackEval"

    factor = py_motmetrics_median / trackeval_median
    return f"TrackEval was {factor:.2f}x faster than py-motmetrics"


def _render_execution_time_table(timings):
    headers = ("Evaluator", *[f"Run {index}" for index in range(1, TIMING_REPEATS + 1)], "Median")
    rows = [
        (
            name,
            *[f"{elapsed:.6f}s" for elapsed in samples],
            f"{statistics.median(samples):.6f}s",
        )
        for name, samples in timings.items()
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render_row(row):
        return " | ".join(value.ljust(width) for value, width in zip(row, widths))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join(
        [
            f"Execution time comparison ({TIMING_REPEATS} warmed runs; lower is better)",
            render_row(headers),
            separator,
            *(render_row(row) for row in rows),
            _relative_time_summary(timings),
        ]
    )


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
            "HOTA, DetA, and AssA values are alpha means; their difference is the maximum over all alphas.",
            *table,
        ]
    )


def _write_github_summary(rows, timings):
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    lines = [
        "## TrackEval parity",
        "",
        f"Maximum permitted absolute difference: `{PARITY_TOLERANCE:.0e}`.",
        "HOTA, DetA, and AssA values are alpha means; their difference is the maximum over all alphas.",
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
    lines.extend(
        [
            "",
            "## Execution time comparison",
            "",
            f"Median of {TIMING_REPEATS} warmed runs on the same in-memory sequences. "
            "This comparison is informational and does not gate CI.",
            "",
            "| Evaluator | Run 1 (s) | Run 2 (s) | Run 3 (s) | Median (s) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, samples in timings.items():
        rendered_samples = " | ".join(f"{elapsed:.6f}" for elapsed in samples)
        lines.append(
            f"| {name} | {rendered_samples} | {statistics.median(samples):.6f} |"
        )
    lines.extend(["", f"**{_relative_time_summary(timings)}.**"])
    with Path(summary_path).open("a", encoding="utf-8") as summary_file:
        summary_file.write("\n".join(lines) + "\n")


def test_metrics_match_trackeval_on_bundled_tud_sequences():
    sequences = {name: _load_sequence(name) for name in SEQUENCE_NAMES}
    py_motmetrics_results, trackeval_results, timings = _measure_execution_times(sequences)

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
    print("\n" + _render_execution_time_table(timings))
    _write_github_summary(rows, timings)

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
