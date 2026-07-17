# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""High-level helpers for common tracker evaluation workflows."""

import multiprocessing
import os
import shutil
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import motmetrics._io as io
import motmetrics._metrics as metrics_module
from motmetrics._accumulator import _Accumulator
from motmetrics._assignment import _linear_sum_assignment
from motmetrics._distances import iou_matrix

HOTA_ALPHAS = np.arange(0.05, 0.99, 0.05)
HOTA_SUMMARY_METRICS = OrderedDict([
    ("hota_alpha", "hota"),
    ("deta_alpha", "deta"),
    ("assa_alpha", "assa"),
])

_WORKER_PROGRESS_CURRENT = None
_WORKER_PROGRESS_TOTAL = None
_WORKER_PROGRESS_STAGE = None

_PROGRESS_WAITING = 0
_PROGRESS_LOADING = 1
_PROGRESS_IOU = 2
_PROGRESS_CLEAR = 3
_PROGRESS_METRICS = 4
_PROGRESS_HOTA = 5
_PROGRESS_DONE = 6
_PROGRESS_FAILED = 7
_PROGRESS_STAGE_NAMES = (
    "waiting",
    "loading",
    "IoU",
    "CLEAR",
    "metrics",
    "HOTA",
    "done",
    "failed",
)
_PROGRESS_FLUSH_FRAMES = 32


class _MOTChallengeSummary(object):
    """MOTChallenge metric results with both dataframe and rendered views."""

    def __init__(self, df, formatters=None, namemap=None):
        self.df = df
        self.formatters = formatters
        self.namemap = namemap

    @property
    def text(self):
        """Human-readable MOTChallenge-style table."""
        return io.render_summary(self.df, formatters=self.formatters, namemap=self.namemap)

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


class _PreparedIoUSequence(object):
    """Per-frame IoU data shared by CLEAR/Identity and HOTA."""

    def __init__(self, frame_data, gt_id_counts, tracker_id_counts, alignment_scores, num_objects, num_predictions):
        self.frame_data = frame_data
        self.gt_id_counts = gt_id_counts
        self.tracker_id_counts = tracker_id_counts
        self.alignment_scores = alignment_scores
        self.num_objects = num_objects
        self.num_predictions = num_predictions


def evaluate_motchallenge(
    groundtruths,
    tests,
    fmt=io.Format.AUTO,
    distfields=None,
    distth=0.5,
    metrics=None,
    name=None,
    generate_overall=True,
    gt_min_confidence=1,
    exclude_id=False,
    include_hota=True,
    hota_alphas=None,
    n_jobs=1,
    progress=None,
):
    """Evaluate MOTChallenge files or folders and return a rich summary.

    Parameters
    ----------
    groundtruths : str or path-like
        Path to a ground-truth file, a sequence folder containing ``gt.txt``, or
        a MOTChallenge root containing ``<sequence>/gt/gt.txt`` files.
    tests : str or path-like
        Path to a tracker result file, a sequence folder containing ``test.txt``,
        or a MOTChallenge tracker root containing ``<sequence>.txt`` files.
    include_hota : bool, optional
        If true, append HOTA, DetA, and AssA averaged over ``hota_alphas``.
    hota_alphas : array-like, optional
        HOTA alpha thresholds. Defaults to the TrackEval thresholds from 0.05 to 0.95.
    n_jobs : int, optional
        Number of sequence worker processes used for folder-based IoU
        evaluation. A single file is evaluated in the calling process. Defaults
        to 1.
    progress : bool, optional
        Display one progress row per sequence. By default this is enabled for
        interactive folder evaluation and disabled when stderr is redirected.

    Returns
    -------
    _MOTChallengeSummary
        Wrapper around the raw pandas DataFrame. ``print(summary)`` displays a
        MOTChallenge-style table, while ``summary.df`` exposes the dataframe.
    """
    gt_path = Path(groundtruths)
    test_path = Path(tests)
    _validate_paths(gt_path, test_path)
    _validate_n_jobs(n_jobs)
    progress_enabled = _progress_is_enabled(progress)

    metric_names = _prepare_metrics(metrics, exclude_id)
    metric_host = metrics_module._METRIC_HOST
    if hota_alphas is None:
        hota_alphas = HOTA_ALPHAS
    summary = _evaluate_iou_paths(
        gt_path,
        test_path,
        name,
        fmt,
        gt_min_confidence,
        distfields,
        distth,
        metric_names,
        include_hota,
        hota_alphas,
        generate_overall and not gt_path.is_file(),
        n_jobs,
        metric_host,
        progress_enabled,
    )

    return _MOTChallengeSummary(
        summary,
        formatters=_summary_formatters(metric_host, include_hota),
        namemap=_summary_namemap(include_hota),
    )


def _evaluate_iou_paths(
    gt_root,
    test_root,
    sequence_name,
    fmt,
    gt_min_confidence,
    distfields,
    distth,
    metric_names,
    include_hota,
    hota_alphas,
    generate_overall,
    n_jobs,
    metric_host,
    progress,
):
    """Evaluate every input through the canonical state-only IoU engine."""
    if gt_root.is_file():
        matched_files = [(sequence_name or _default_sequence_name(test_root), gt_root, test_root)]
    else:
        gt_files = _find_groundtruth_files(gt_root)
        test_files = _find_test_files(test_root)
        matched_files = [
            (name, gt_files[name], test_path)
            for name, test_path in test_files.items()
            if name in gt_files
        ]
    tasks = [
        (
            task_index,
            name,
            gt_path,
            test_path,
            fmt,
            gt_min_confidence,
            distfields,
            distth,
            metric_names,
            include_hota,
            np.asarray(hota_alphas, dtype=float),
        )
        for task_index, (name, gt_path, test_path) in enumerate(matched_files)
    ]
    if not tasks:
        raise ValueError("No matching ground-truth and tracker result files found.")

    workers = min(n_jobs, len(tasks))
    context = _fast_process_context()
    progress_arrays = _create_progress_arrays(context, len(tasks)) if progress else (None, None, None)
    _initialize_evaluation_worker(*progress_arrays)
    progress_names = [task[1] for task in tasks]
    display = _SequenceProgressDisplay(progress_names, progress_arrays, enabled=progress)
    if workers == 1:
        with display:
            results = [_evaluate_iou_sequence_file(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_evaluation_worker,
            initargs=progress_arrays,
        ) as executor:
            # Submitting before the renderer thread starts ensures POSIX workers
            # are forked from a single-threaded parent.
            futures = [executor.submit(_evaluate_iou_sequence_file, task) for task in tasks]
            with display:
                results = [future.result() for future in futures]

    names = [result[0] for result in results]
    partials = [result[1] for result in results]
    rows = [OrderedDict((metric, partial[metric]) for metric in metric_names) for partial in partials]
    result_names = list(names)
    if generate_overall:
        rows.append(
            metric_host.compute_overall(
                partials,
                metrics=metric_names,
            )
        )
        result_names.append("OVERALL")
    summary = pd.DataFrame(rows, index=result_names, columns=metric_names)

    if include_hota:
        sequence_summaries = OrderedDict((result[0], result[2]) for result in results)
        if generate_overall:
            sequence_summaries["OVERALL"] = _combine_hota_sequence_summaries(
                sequence_summaries.values()
            )
        summary = pd.concat(
            [summary, _hota_summary_frame(sequence_summaries, summary.index)],
            axis=1,
        )
    return summary


def _evaluate_iou_sequence_file(task):
    """Load, match, and summarize one sequence inside a worker process."""
    (
        task_index,
        name,
        gt_path,
        test_path,
        fmt,
        gt_min_confidence,
        distfields,
        distth,
        metric_names,
        include_hota,
        hota_alphas,
    ) = task
    sequence_progress = _WorkerSequenceProgress(task_index, include_hota) if _WORKER_PROGRESS_STAGE is not None else None
    if sequence_progress is not None:
        sequence_progress.stage(_PROGRESS_LOADING)
    try:
        ground_truth = io.loadtxt(gt_path, fmt=fmt, min_confidence=gt_min_confidence)
        tracker = io.loadtxt(test_path, fmt=fmt)
        prepared = _prepare_iou_sequence_data(ground_truth, tracker, distfields, progress=sequence_progress)
        if sequence_progress is not None:
            sequence_progress.stage(_PROGRESS_CLEAR)
        accumulator = _compare_prepared_iou(prepared, distth, progress=sequence_progress)

        if sequence_progress is not None:
            sequence_progress.stage(_PROGRESS_METRICS)
        partial = metrics_module._METRIC_HOST.compute(
            accumulator,
            metrics=metric_names,
        )
        if include_hota:
            if sequence_progress is not None:
                sequence_progress.stage(_PROGRESS_HOTA)
            hota_summary = _compute_prepared_hota_sequence_summary(
                prepared,
                hota_alphas,
                progress=sequence_progress,
            )
        else:
            hota_summary = None
        if sequence_progress is not None:
            sequence_progress.finish()
        return name, partial, hota_summary
    except BaseException:
        if sequence_progress is not None:
            sequence_progress.fail()
        raise


def _initialize_evaluation_worker(
    progress_current=None,
    progress_total=None,
    progress_stage=None,
):
    """Connect each sequence process to the shared progress arrays."""
    global _WORKER_PROGRESS_CURRENT, _WORKER_PROGRESS_TOTAL, _WORKER_PROGRESS_STAGE  # pylint: disable=global-statement
    _WORKER_PROGRESS_CURRENT = progress_current
    _WORKER_PROGRESS_TOTAL = progress_total
    _WORKER_PROGRESS_STAGE = progress_stage


class _WorkerSequenceProgress(object):
    """Publish batched frame counters from one sequence worker."""

    def __init__(self, index, include_hota):
        self.index = index
        self.passes = 3 if include_hota else 2
        self.current = 0
        self.total = 0

    def begin(self, frame_count):
        """Set the frame-derived total and enter the IoU pass."""
        self.total = frame_count * self.passes
        _WORKER_PROGRESS_TOTAL[self.index] = self.total
        self.stage(_PROGRESS_IOU)

    def advance(self):
        """Advance locally, publishing only occasional cache-friendly writes."""
        self.current += 1
        if self.current % _PROGRESS_FLUSH_FRAMES == 0:
            _WORKER_PROGRESS_CURRENT[self.index] = self.current

    def stage(self, stage):
        """Flush the counter and publish a new processing stage."""
        _WORKER_PROGRESS_CURRENT[self.index] = self.current
        _WORKER_PROGRESS_STAGE[self.index] = stage

    def finish(self):
        """Mark the sequence complete."""
        self.current = self.total
        self.stage(_PROGRESS_DONE)

    def fail(self):
        """Mark the sequence failed before propagating its exception."""
        self.stage(_PROGRESS_FAILED)


class _SequenceProgressDisplay(object):
    """Render all worker counters from one parent-owned terminal thread."""

    def __init__(self, names, arrays, enabled, stream=None):
        self.names = names
        self.current, self.total, self.stage = arrays
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self.stop_event = threading.Event()
        self.thread = None
        self.name_width = min(30, max(len(name) for name in names))
        columns = shutil.get_terminal_size(fallback=(80, 24)).columns
        self.bar_width = max(8, min(30, columns - self.name_width - 22))

    def __enter__(self):
        if not self.enabled:
            return self
        self.stream.write("\x1b[?25l" + "\n" * len(self.names))
        self._render()
        self.thread = threading.Thread(target=self._run, name="motmetrics-progress", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return False
        self.stop_event.set()
        self.thread.join()
        self._render()
        self.stream.write("\x1b[?25h")
        self.stream.flush()
        return False

    def _run(self):
        while not self.stop_event.wait(0.1):
            self._render()

    def _render(self):
        lines = [
            _format_progress_line(
                name,
                self.current[index],
                self.total[index],
                self.stage[index],
                self.name_width,
                self.bar_width,
            )
            for index, name in enumerate(self.names)
        ]
        output = "\x1b[{}A".format(len(lines))
        output += "".join("\r\x1b[2K{}\n".format(line) for line in lines)
        self.stream.write(output)
        self.stream.flush()


def _format_progress_line(name, current, total, stage, name_width, bar_width):
    """Format a fixed-width sequence progress row."""
    if len(name) > name_width:
        name = name[: max(0, name_width - 1)] + "…"
    label = name.ljust(name_width)
    if stage == _PROGRESS_DONE:
        fraction = 1.0
    elif total:
        fraction = min(1.0, current / total)
    else:
        fraction = 0.0
    filled = int(bar_width * fraction)
    bar = "█" * filled + "·" * (bar_width - filled)
    percent = "{:3.0f}%".format(fraction * 100) if total or stage == _PROGRESS_DONE else " --%"
    stage_name = _PROGRESS_STAGE_NAMES[stage]
    return "{} [{}] {} {}".format(label, bar, percent, stage_name)


def _create_progress_arrays(context, count):
    """Allocate lock-free counters inherited by all sequence workers."""
    return context.RawArray("q", count), context.RawArray("q", count), context.RawArray("b", count)


def _progress_is_enabled(progress):
    """Resolve automatic terminal progress without polluting redirected logs."""
    if progress is not None and not isinstance(progress, (bool, np.bool_)):
        raise TypeError("progress must be True, False, or None.")
    if progress is not None:
        return bool(progress)
    is_terminal = callable(getattr(sys.stderr, "isatty", None)) and sys.stderr.isatty()
    return is_terminal and os.environ.get("TERM") != "dumb"


def _fast_process_context():
    """Prefer copy-on-write workers on POSIX to avoid dataframe serialization."""
    if os.name == "posix" and "fork" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context()


def _hota_summary_frame(sequence_summaries, index):
    """Convert per-alpha sequence results into displayed scalar means."""
    rows = []
    for row_name in index:
        row = OrderedDict()
        for alpha_metric, summary_metric in HOTA_SUMMARY_METRICS.items():
            row[summary_metric] = np.mean(sequence_summaries[row_name][alpha_metric])
        rows.append(row)
    return pd.DataFrame(rows, index=index)


def _compute_prepared_hota_sequence_summary(prepared, hota_alphas, progress=None):
    frame_data = prepared.frame_data
    gt_id_counts = prepared.gt_id_counts
    tracker_id_counts = prepared.tracker_id_counts
    alignment_scores = prepared.alignment_scores
    num_objects = prepared.num_objects
    num_predictions = prepared.num_predictions
    num_alphas = len(hota_alphas)
    num_gt_ids = len(gt_id_counts)
    num_tracker_ids = len(tracker_id_counts)
    true_positives = np.zeros(num_alphas)
    match_counts = np.zeros((num_alphas, num_gt_ids, num_tracker_ids))
    matched_gt_indices = []
    matched_tracker_indices = []
    matched_similarities = []

    for _, gt_indices, tracker_indices, similarities in frame_data:
        if progress is not None:
            progress.advance()
        if similarities.size == 0:
            continue
        weighted_similarities = similarities * alignment_scores[np.ix_(gt_indices, tracker_indices)]
        row_indices, col_indices = _linear_sum_assignment(1 - weighted_similarities)
        if len(row_indices) == 0:
            continue
        matched_gt_indices.append(gt_indices[row_indices])
        matched_tracker_indices.append(tracker_indices[col_indices])
        matched_similarities.append(similarities[row_indices, col_indices])

    if matched_similarities:
        matched_gt_indices = np.concatenate(matched_gt_indices)
        matched_tracker_indices = np.concatenate(matched_tracker_indices)
        matched_similarities = np.concatenate(matched_similarities)
        valid_matches = matched_similarities[:, np.newaxis] >= hota_alphas - np.finfo("float").eps
        true_positives = valid_matches.sum(axis=0, dtype=float)
        pair_indices = matched_gt_indices * num_tracker_ids + matched_tracker_indices
        num_id_pairs = num_gt_ids * num_tracker_ids
        for alpha_index in range(num_alphas):
            match_counts[alpha_index] = np.bincount(
                pair_indices[valid_matches[:, alpha_index]],
                minlength=num_id_pairs,
            ).reshape(num_gt_ids, num_tracker_ids)

    false_positives = num_predictions - true_positives
    deta = _quiet_divide(true_positives, np.maximum(1, num_objects + false_positives))
    assa = _compute_hota_assa(match_counts, gt_id_counts, tracker_id_counts, true_positives)
    hota = np.sqrt(deta * assa)
    return {
        "hota_alpha": hota,
        "deta_alpha": deta,
        "assa_alpha": assa,
        "num_detections": true_positives,
        "num_objects": num_objects,
        "num_false_positives": false_positives,
    }


def _prepare_iou_sequence_data(gt, test, distfields=None, progress=None):
    if distfields is None:
        distfields = ["X", "Y", "Width", "Height"]

    gt = gt[distfields]
    test = test[distfields]
    gt_ids = pd.Index(np.sort(gt.index.get_level_values("Id").unique()))
    tracker_ids = pd.Index(np.sort(test.index.get_level_values("Id").unique()))
    gt_groups, gt_id_counts = _group_frame_arrays(gt, gt_ids)
    test_groups, tracker_id_counts = _group_frame_arrays(test, tracker_ids)
    frame_ids = pd.Index(gt_groups).union(pd.Index(test_groups)).sort_values()

    potential_matches = np.zeros((len(gt_ids), len(tracker_ids)))
    frame_data = []
    empty_indices = np.empty(0, dtype=int)
    empty_values = np.empty((0, len(distfields)), dtype=float)
    if progress is not None:
        progress.begin(len(frame_ids))

    for frame_id in frame_ids:
        if progress is not None:
            progress.advance()
        frame_gt_indices, frame_gt_values = gt_groups.get(frame_id, (empty_indices, empty_values))
        frame_tracker_indices, frame_tracker_values = test_groups.get(frame_id, (empty_indices, empty_values))
        similarities = iou_matrix(frame_gt_values, frame_tracker_values, return_dist=False)
        frame_data.append((frame_id, frame_gt_indices, frame_tracker_indices, similarities))
        if similarities.size == 0:
            continue

        similarity_denominator = similarities.sum(0)[np.newaxis, :] + similarities.sum(1)[:, np.newaxis] - similarities
        similarity_iou = np.zeros_like(similarities)
        similarity_mask = similarity_denominator > 0 + np.finfo("float").eps
        similarity_iou[similarity_mask] = similarities[similarity_mask] / similarity_denominator[similarity_mask]
        potential_matches[np.ix_(frame_gt_indices, frame_tracker_indices)] += similarity_iou

    alignment_scores = _quiet_divide(
        potential_matches,
        np.maximum(1, gt_id_counts[:, np.newaxis] + tracker_id_counts[np.newaxis, :] - potential_matches),
    )
    return _PreparedIoUSequence(
        frame_data=frame_data,
        gt_id_counts=gt_id_counts,
        tracker_id_counts=tracker_id_counts,
        alignment_scores=alignment_scores,
        num_objects=len(gt),
        num_predictions=len(test),
    )


def _group_frame_arrays(dataframe, id_index):
    """Group MOT rows into array views with one dataframe conversion."""
    frame_ids = dataframe.index.get_level_values("FrameId").to_numpy()
    id_codes = id_index.get_indexer(dataframe.index.get_level_values("Id"))
    values = dataframe.to_numpy(dtype=float, copy=False)
    if len(frame_ids) == 0:
        return {}, np.zeros(len(id_index))

    if np.any(frame_ids[1:] < frame_ids[:-1]):
        order = np.argsort(frame_ids, kind="stable")
        frame_ids = frame_ids[order]
        id_codes = id_codes[order]
        values = values[order]

    boundaries = np.flatnonzero(frame_ids[1:] != frame_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(frame_ids)]))
    groups = {}
    id_counts = np.zeros(len(id_index))
    for start, end in zip(starts, ends):
        frame_id_codes = id_codes[start:end]
        groups[frame_ids[start]] = (frame_id_codes, values[start:end])
        np.add.at(id_counts, frame_id_codes, 1)
    return groups, id_counts


def _compare_prepared_iou(prepared, distth, progress=None):
    accumulator = _Accumulator()
    for frame_id, gt_indices, tracker_indices, similarities in prepared.frame_data:
        if progress is not None:
            progress.advance()
        distances = 1 - similarities
        distances = np.where(distances > distth, np.nan, distances)
        accumulator.update(gt_indices, tracker_indices, distances, frameid=frame_id)
    return accumulator


def _compute_hota_assa(match_counts, gt_id_counts, tracker_id_counts, true_positives):
    if match_counts.shape[1] == 0 or match_counts.shape[2] == 0:
        return _quiet_divide(np.zeros_like(true_positives), np.maximum(1, true_positives))
    assa_denominator = gt_id_counts[np.newaxis, :, np.newaxis] + tracker_id_counts[np.newaxis, np.newaxis, :] - match_counts
    assa_per_pair = _quiet_divide(match_counts, np.maximum(1, assa_denominator))
    return _quiet_divide((assa_per_pair * match_counts).sum(axis=(1, 2)), np.maximum(1, true_positives))


def _combine_hota_sequence_summaries(summaries):
    summaries = list(summaries)
    true_positives = np.sum([summary["num_detections"] for summary in summaries], axis=0)
    num_objects = sum(summary["num_objects"] for summary in summaries)
    false_positives = np.sum([summary["num_false_positives"] for summary in summaries], axis=0)
    deta = _quiet_divide(true_positives, np.maximum(1, num_objects + false_positives))
    assa = _quiet_divide(
        np.sum([summary["assa_alpha"] * summary["num_detections"] for summary in summaries], axis=0),
        np.maximum(1, true_positives),
    )
    hota = np.sqrt(deta * assa)
    return {
        "hota_alpha": hota,
        "deta_alpha": deta,
        "assa_alpha": assa,
        "num_detections": true_positives,
        "num_objects": num_objects,
        "num_false_positives": false_positives,
    }


def _quiet_divide(numerator, denominator):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.true_divide(numerator, denominator)


def _prepare_metrics(metric_names, exclude_id):
    if metric_names is None:
        metric_names = list(metrics_module.motchallenge_metrics)
    elif isinstance(metric_names, str):
        metric_names = [metric_names]
    else:
        metric_names = list(metric_names)

    if exclude_id:
        metric_names = [metric_name for metric_name in metric_names if not metric_name.startswith("id")]
    return metric_names


def _summary_formatters(metric_host, include_hota):
    formatters = dict(metric_host.formatters)
    if include_hota:
        formatters.update({"hota": "{:.1%}".format, "deta": "{:.1%}".format, "assa": "{:.1%}".format})
    return formatters


def _summary_namemap(include_hota):
    namemap = dict(io.motchallenge_metric_names)
    if include_hota:
        namemap.update({"hota": "HOTA", "deta": "DetA", "assa": "AssA"})
    return namemap


def _validate_paths(gt_path, test_path):
    if not gt_path.exists():
        raise FileNotFoundError("Ground-truth path not found: {}".format(gt_path))
    if not test_path.exists():
        raise FileNotFoundError("Tracker result path not found: {}".format(test_path))
    if gt_path.is_file() != test_path.is_file():
        raise ValueError("Ground-truth and tracker result paths must both be files or both be folders.")


def _validate_n_jobs(n_jobs):
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1.")


def _default_sequence_name(path):
    path = Path(path)
    if path.stem in ("gt", "test"):
        return path.parent.name
    return path.stem


def _find_groundtruth_files(root):
    files = OrderedDict()
    direct = root / "gt.txt"
    if direct.is_file():
        files[root.name] = direct

    for path in sorted(root.glob("*/gt/gt.txt")):
        files.setdefault(path.parents[1].name, path)
    for path in sorted(root.glob("*/gt.txt")):
        files.setdefault(path.parent.name, path)
    return files


def _find_test_files(root):
    files = OrderedDict()
    direct = root / "test.txt"
    if direct.is_file():
        files[root.name] = direct

    for path in sorted(root.glob("*.txt")):
        if path.name.startswith("eval") or path.name in ("gt.txt", "test.txt"):
            continue
        files.setdefault(path.stem, path)
    for path in sorted(root.glob("*/test.txt")):
        files.setdefault(path.parent.name, path)
    return files
