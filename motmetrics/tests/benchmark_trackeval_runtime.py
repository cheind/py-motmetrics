"""Compare complete py-motmetrics and TrackEval runtimes in fresh processes."""

import sys
from pathlib import Path

DATA_DIR = Path(__file__).parents[1] / "data"
SEQUENCE_NAMES = ("TUD-Campus", "TUD-Stadtmitte")


def _run_motmetrics():
    import motmetrics as mm

    summary = mm.evaluate_motchallenge(
        DATA_DIR,
        DATA_DIR,
        n_jobs=1,
        progress=False,
    )
    if summary.index != [*SEQUENCE_NAMES, "OVERALL"]:
        raise RuntimeError("py-motmetrics did not produce every expected result.")


def _run_trackeval():
    import numpy as np
    import trackeval

    metric_objects = {
        "HOTA": trackeval.metrics.HOTA(),
        "CLEAR": trackeval.metrics.CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
        "Identity": trackeval.metrics.Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
    }
    sequence_results = {}
    for sequence_name in SEQUENCE_NAMES:
        sequence_dir = DATA_DIR / sequence_name
        data = _load_trackeval_sequence(
            sequence_dir / "gt.txt",
            sequence_dir / "test.txt",
            np,
        )
        sequence_results[sequence_name] = {
            metric_name: metric.eval_sequence(data) for metric_name, metric in metric_objects.items()
        }

    combined = {
        metric_name: metric.combine_sequences(
            {sequence_name: sequence_results[sequence_name][metric_name] for sequence_name in SEQUENCE_NAMES}
        )
        for metric_name, metric in metric_objects.items()
    }
    if not all(combined.values()):
        raise RuntimeError("TrackEval did not produce every expected result.")


def _load_trackeval_sequence(ground_truth_path, tracker_path, np):
    ground_truth = np.loadtxt(ground_truth_path, delimiter=",", ndmin=2)
    tracker = np.loadtxt(tracker_path, delimiter=",", ndmin=2)
    ground_truth = ground_truth[ground_truth[:, 6] >= 1]

    ground_truth_ids = {value: index for index, value in enumerate(np.unique(ground_truth[:, 1].astype(np.int64)))}
    tracker_ids = {value: index for index, value in enumerate(np.unique(tracker[:, 1].astype(np.int64)))}
    frame_ids = np.union1d(ground_truth[:, 0], tracker[:, 0])

    gt_ids = []
    tracker_ids_by_frame = []
    similarity_scores = []
    for frame_id in frame_ids:
        frame_ground_truth = ground_truth[ground_truth[:, 0] == frame_id]
        frame_tracker = tracker[tracker[:, 0] == frame_id]
        gt_ids.append(
            np.asarray(
                [ground_truth_ids[value] for value in frame_ground_truth[:, 1].astype(np.int64)],
                dtype=int,
            )
        )
        tracker_ids_by_frame.append(
            np.asarray(
                [tracker_ids[value] for value in frame_tracker[:, 1].astype(np.int64)],
                dtype=int,
            )
        )
        similarity_scores.append(_box_iou(frame_ground_truth[:, 2:6], frame_tracker[:, 2:6], np))

    return {
        "num_timesteps": len(frame_ids),
        "num_gt_ids": len(ground_truth_ids),
        "num_tracker_ids": len(tracker_ids),
        "num_gt_dets": len(ground_truth),
        "num_tracker_dets": len(tracker),
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids_by_frame,
        "similarity_scores": similarity_scores,
    }


def _box_iou(left, right, np):
    if len(left) == 0 or len(right) == 0:
        return np.empty((len(left), len(right)), dtype=float)

    left_max = left[:, :2] + left[:, 2:]
    right_max = right[:, :2] + right[:, 2:]
    intersection_size = np.maximum(
        0,
        np.minimum(left_max[:, np.newaxis], right_max[np.newaxis, :])
        - np.maximum(left[:, np.newaxis, :2], right[np.newaxis, :, :2]),
    )
    intersection = intersection_size.prod(axis=2)
    union = left[:, 2:].prod(axis=1)[:, np.newaxis] + right[:, 2:].prod(axis=1)[np.newaxis, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def _measure_fresh_processes(runs):
    import subprocess
    import time

    samples = {"py-motmetrics": [], "TrackEval": []}
    modes = {"py-motmetrics": "motmetrics", "TrackEval": "trackeval"}
    script_path = str(Path(__file__).resolve())
    for run_index in range(runs):
        order = tuple(samples) if run_index % 2 == 0 else tuple(reversed(samples))
        for backend in order:
            started = time.perf_counter()
            subprocess.run(
                [sys.executable, script_path, "--backend", modes[backend]],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            samples[backend].append(time.perf_counter() - started)
    return samples


def _render_timing_summary(samples):
    import statistics

    statistics_by_backend = {
        backend: {
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
        }
        for backend, values in samples.items()
    }
    ratio = statistics_by_backend["TrackEval"]["median"] / statistics_by_backend["py-motmetrics"]["median"]
    lines = [
        "## End-to-end metrics runtime",
        "",
        "Each sample uses a fresh Python process and includes interpreter startup, imports, file loading, IoU preparation, and all CLEAR, Identity, and HOTA calculations.",
        "",
        "| Backend | Median | Minimum | Maximum | Fresh processes |",
        "|---|---:|---:|---:|---:|",
    ]
    for backend, values in samples.items():
        backend_statistics = statistics_by_backend[backend]
        lines.append(
            "| {} | {:.3f} s | {:.3f} s | {:.3f} s | {} |".format(
                backend,
                backend_statistics["median"],
                backend_statistics["minimum"],
                backend_statistics["maximum"],
                len(values),
            )
        )
    lines.extend(
        [
            "",
            "TrackEval / py-motmetrics median runtime ratio: **{:.2f}x**.".format(ratio),
        ]
    )
    return "\n".join(lines)


def _write_github_summary(rendered):
    import os

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary_file:
            summary_file.write(rendered + "\n")


def main():
    if sys.argv[1:] == ["--backend", "motmetrics"]:
        _run_motmetrics()
        return
    if sys.argv[1:] == ["--backend", "trackeval"]:
        _run_trackeval()
        return

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=7)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    samples = _measure_fresh_processes(args.runs)
    rendered = _render_timing_summary(samples)
    print(rendered)
    _write_github_summary(rendered)


if __name__ == "__main__":
    main()
