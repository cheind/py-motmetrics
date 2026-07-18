# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""High-level helpers for common tracker evaluation workflows."""

import os
import sys
from collections import OrderedDict
from collections.abc import Collection, Mapping, Sequence
from numbers import Number
from pathlib import Path

import numpy as np

import motmetrics._io as io
import motmetrics._metrics as metrics_module
from motmetrics._accumulator import _Accumulator
from motmetrics._assignment import _dense_linear_sum_assignment
from motmetrics._distances import iou_matrix
from motmetrics._extensions import (
    SUPPORTED_INTERMEDIATES,
    MetricContext,
    MetricFamily,
    SequenceView,
    _ClearEventRecorder,
)

HOTA_ALPHAS = np.arange(0.05, 0.99, 0.05)
_FLOAT_EPS = np.finfo(float).eps
HOTA_SUMMARY_METRICS = OrderedDict([
    ("hota_alpha", "hota"),
    ("deta_alpha", "deta"),
    ("assa_alpha", "assa"),
    ("detre_alpha", "detre"),
    ("detpr_alpha", "detpr"),
    ("assre_alpha", "assre"),
    ("asspr_alpha", "asspr"),
    ("loca_alpha", "loca"),
    ("owta_alpha", "owta"),
])

_WORKER_PROGRESS_CURRENT = None
_WORKER_PROGRESS_TOTAL = None
_WORKER_PROGRESS_STAGE = None

_PROGRESS_WAITING = 0
_PROGRESS_LOADING = 1
_PROGRESS_IOU = 2
_PROGRESS_METRICS = 3
_PROGRESS_HOTA = 4
_PROGRESS_EXTENSIONS = 5
_PROGRESS_DONE = 6
_PROGRESS_FAILED = 7
_PROGRESS_STAGE_NAMES = (
    "waiting",
    "loading",
    "IoU+CLEAR",
    "metrics",
    "HOTA",
    "extensions",
    "done",
    "failed",
)
_PROGRESS_FLUSH_FRAMES = 32


class _MOTChallengeSummary(object):
    """MOTChallenge results with a native text view and lazy dataframe."""

    def __init__(self, rows, index, columns, formatters=None, namemap=None):
        self._rows = rows
        self.index = index
        self.columns = columns
        self.formatters = formatters
        self.namemap = namemap
        self._df = None

    def __getitem__(self, key):
        """Return one metric value or an ordered metric column without pandas."""
        if isinstance(key, tuple):
            if len(key) != 2:
                raise TypeError("Summary tuple keys must contain (row, metric).")
            row_name, metric_name = key
            if metric_name not in self.columns:
                raise KeyError("Unknown summary metric: {!r}".format(metric_name))
            try:
                row_position = self.index.index(row_name)
            except ValueError:
                raise KeyError("Unknown summary row: {!r}".format(row_name)) from None
            return self._rows[row_position][metric_name]

        if isinstance(key, str):
            if key in self.columns:
                return OrderedDict(
                    (row_name, row[key])
                    for row_name, row in zip(self.index, self._rows)
                )
            if key in self.index:
                row_position = self.index.index(key)
                row = self._rows[row_position]
                return OrderedDict((metric_name, row[metric_name]) for metric_name in self.columns)
            raise KeyError("Unknown summary row or metric: {!r}".format(key))

        raise TypeError("Summary keys must be a row, metric name, or a (row, metric) pair.")

    @property
    def df(self):
        """Materialize a pandas dataframe only when explicitly requested."""
        if self._df is None:
            import pandas as pd

            self._df = pd.DataFrame(self._rows, index=self.index, columns=self.columns)
        return self._df

    @property
    def text(self):
        """Human-readable MOTChallenge-style table."""
        return io.render_summary(
            self._rows,
            self.index,
            self.columns,
            formatters=self.formatters,
            namemap=self.namemap,
        )

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


class _PreparedIoUSequence(object):
    """Per-frame IoU data shared by CLEAR/Identity and HOTA."""

    def __init__(
        self,
        frame_data,
        gt_id_counts,
        tracker_id_counts,
        alignment_scores,
        frame_ids,
        ground_truth_ids,
        tracker_ids,
        num_objects,
        num_predictions,
        accumulator,
    ):
        self.frame_data = frame_data
        self.gt_id_counts = gt_id_counts
        self.tracker_id_counts = tracker_id_counts
        self.alignment_scores = alignment_scores
        self.frame_ids = frame_ids
        self.ground_truth_ids = ground_truth_ids
        self.tracker_ids = tracker_ids
        self.num_objects = num_objects
        self.num_predictions = num_predictions
        self.accumulator = accumulator


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
    hota_alphas=None,
    n_jobs=1,
    progress=None,
    extra_metric_families=None,
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
    hota_alphas : array-like, optional
        HOTA alpha thresholds. Defaults to the TrackEval thresholds from 0.05 to 0.95.
    n_jobs : int, optional
        Number of sequence worker processes used for folder-based IoU
        evaluation. A single file is evaluated in the calling process. Defaults
        to 1.
    progress : bool, optional
        Display one progress row per sequence. By default this is enabled for
        interactive folder evaluation and disabled when stderr is redirected.
    extra_metric_families : MetricFamily or iterable of MetricFamily, optional
        Explicit, per-call metric extensions. Each family owns its matching or
        aggregation semantics and declares any opt-in intermediate data it
        needs. Family instances must be picklable when ``n_jobs > 1``.

    Returns
    -------
    _MOTChallengeSummary
        Native MOTChallenge-style results. ``print(summary)`` does not import
        pandas; ``summary.df`` materializes a dataframe on demand.
    """
    gt_path = Path(groundtruths)
    test_path = Path(tests)
    _validate_paths(gt_path, test_path)
    _validate_n_jobs(n_jobs)
    progress_enabled = _progress_is_enabled(progress)

    metric_names = _prepare_metrics(metrics, exclude_id)
    metric_families = _prepare_metric_families(
        extra_metric_families,
        n_jobs,
    )
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
        hota_alphas,
        generate_overall and not gt_path.is_file(),
        n_jobs,
        progress_enabled,
        metric_families,
    )

    return _MOTChallengeSummary(
        *summary,
        formatters=_summary_formatters(metric_families),
        namemap=_summary_namemap(metric_families),
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
    hota_alphas,
    generate_overall,
    n_jobs,
    progress,
    metric_families,
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
            np.asarray(hota_alphas, dtype=float),
            metric_families,
        )
        for task_index, (name, gt_path, test_path) in enumerate(matched_files)
    ]
    if not tasks:
        raise ValueError("No matching ground-truth and tracker result files found.")

    workers = min(n_jobs, len(tasks))
    context = _fast_process_context() if workers > 1 or progress else None
    progress_arrays = _create_progress_arrays(context, len(tasks)) if progress else (None, None, None)
    _initialize_evaluation_worker(*progress_arrays)
    progress_names = [task[1] for task in tasks]
    display = _SequenceProgressDisplay(progress_names, progress_arrays, enabled=progress)
    if workers == 1:
        with display:
            results = [_evaluate_iou_sequence_file(task) for task in tasks]
    else:
        with context.Pool(
            processes=workers,
            initializer=_initialize_evaluation_worker,
            initargs=progress_arrays,
        ) as pool:
            # Submitting before the renderer thread starts ensures POSIX workers
            # are forked from a single-threaded parent.
            pending_results = pool.map_async(_evaluate_iou_sequence_file, tasks)
            with display:
                results = pending_results.get()

    names = [result[0] for result in results]
    partials = [result[1] for result in results]
    rows = [OrderedDict((metric, partial[metric]) for metric in metric_names) for partial in partials]
    result_names = list(names)
    if generate_overall:
        rows.append(
            metrics_module._compute_overall(
                partials,
                metric_names=metric_names,
            )
        )
        result_names.append("OVERALL")
    sequence_summaries = OrderedDict((result[0], result[2]) for result in results)
    if generate_overall:
        sequence_summaries["OVERALL"] = _combine_hota_sequence_summaries(
            sequence_summaries.values()
        )
    for row_name, row in zip(result_names, rows):
        row.update(
            (summary_metric, np.mean(sequence_summaries[row_name][alpha_metric]))
            for alpha_metric, summary_metric in HOTA_SUMMARY_METRICS.items()
        )
    _extend_rows_with_metric_families(
        rows,
        results,
        generate_overall,
        metric_families,
    )
    columns = list(metric_names) + list(HOTA_SUMMARY_METRICS.values())
    for family in metric_families:
        columns.extend(family.metric_names)
    return rows, result_names, columns


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
        hota_alphas,
        metric_families,
    ) = task
    sequence_progress = _WorkerSequenceProgress(task_index) if _WORKER_PROGRESS_STAGE is not None else None
    if sequence_progress is not None:
        sequence_progress.stage(_PROGRESS_LOADING)
    try:
        ground_truth = io.loadtxt(gt_path, fmt=fmt, min_confidence=gt_min_confidence)
        tracker = io.loadtxt(test_path, fmt=fmt)
        event_recorder = _create_clear_event_recorder(
            metric_families,
            ground_truth,
            tracker,
        )
        prepared = _prepare_iou_sequence_data(
            ground_truth,
            tracker,
            distth,
            distfields,
            progress=sequence_progress,
            event_recorder=event_recorder,
        )

        if sequence_progress is not None:
            sequence_progress.stage(_PROGRESS_METRICS)
        partial = metrics_module._compute_metrics(
            prepared.accumulator,
            metric_names=metric_names,
        )
        if sequence_progress is not None:
            sequence_progress.stage(_PROGRESS_HOTA)
        hota_summary = _compute_prepared_hota_sequence_summary(
            prepared,
            hota_alphas,
            progress=sequence_progress,
        )
        family_results = ()
        if metric_families:
            if sequence_progress is not None:
                sequence_progress.stage(_PROGRESS_EXTENSIONS)
            family_results = _evaluate_metric_families(
                name,
                ground_truth,
                tracker,
                prepared,
                event_recorder,
                metric_families,
            )
        if sequence_progress is not None:
            sequence_progress.finish()
        return name, partial, hota_summary, family_results
    except BaseException:
        if sequence_progress is not None:
            sequence_progress.fail()
        raise


def _create_clear_event_recorder(metric_families, ground_truth, tracker):
    if not metric_families:
        return None
    if not any(
        "clear_events" in family.requirements
        for family in metric_families
    ):
        return None
    return _ClearEventRecorder(
        np.unique(ground_truth.ids),
        np.unique(tracker.ids),
    )


def _evaluate_metric_families(
    name,
    ground_truth,
    tracker,
    prepared,
    event_recorder,
    metric_families,
):
    sequence = SequenceView(name, ground_truth, tracker)
    intermediates = MetricContext(
        sequence,
        prepared,
        event_recorder.finish() if event_recorder is not None else None,
    )
    results = []
    for family in metric_families:
        family_intermediates = intermediates._for_requirements(
            family.requirements
        )
        partial = family.evaluate_sequence(sequence, family_intermediates)
        values = _validate_metric_family_values(
            family,
            family.summarize(partial),
            name,
        )
        results.append((partial, values))
    return tuple(results)


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

    def __init__(self, index):
        self.index = index
        self.current = 0
        self.total = 0

    def begin(self, frame_count):
        """Set the frame-derived total and enter the IoU pass."""
        self.total = frame_count * 2
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
        self.thread = None
        self.name_width = min(30, max(len(name) for name in names))
        self.stop_event = None
        self.bar_width = 8
        if enabled:
            import shutil
            import threading

            self.stop_event = threading.Event()
            columns = shutil.get_terminal_size(fallback=(80, 24)).columns
            self.bar_width = max(8, min(30, columns - self.name_width - 22))

    def __enter__(self):
        if not self.enabled:
            return self
        import threading

        self.stream.write("\x1b[?25l" + "\n" * len(self.names))
        self._render()
        self.thread = threading.Thread(target=self._run, name="motmetrics-progress", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
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
    """Prefer copy-on-write workers on POSIX for low startup overhead."""
    import multiprocessing

    if os.name == "posix" and "fork" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context()


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
    localization_sums = np.zeros(num_alphas)
    match_alpha_indices = np.empty(0, dtype=np.intp)
    match_gt_indices = np.empty(0, dtype=np.intp)
    match_tracker_indices = np.empty(0, dtype=np.intp)
    match_counts = np.empty(0)
    matched_gt_indices = []
    matched_tracker_indices = []
    matched_similarities = []

    for gt_indices, tracker_indices, similarities in frame_data:
        if progress is not None:
            progress.advance()
        if similarities.size == 0:
            continue
        weighted_similarities = similarities * alignment_scores[gt_indices[:, None], tracker_indices]
        row_indices, col_indices = _dense_linear_sum_assignment(1 - weighted_similarities)
        if len(row_indices) == 0:
            continue
        matched_gt_indices.append(gt_indices[row_indices])
        matched_tracker_indices.append(tracker_indices[col_indices])
        matched_similarities.append(similarities[row_indices, col_indices])

    if matched_similarities:
        matched_gt_indices = np.concatenate(matched_gt_indices)
        matched_tracker_indices = np.concatenate(matched_tracker_indices)
        matched_similarities = np.concatenate(matched_similarities)
        valid_matches = matched_similarities[:, np.newaxis] >= hota_alphas - _FLOAT_EPS
        true_positives = valid_matches.sum(axis=0, dtype=float)
        localization_sums = matched_similarities @ valid_matches
        pair_indices = matched_gt_indices * num_tracker_ids + matched_tracker_indices
        num_id_pairs = num_gt_ids * num_tracker_ids
        alpha_indices, match_indices = np.nonzero(valid_matches.T)
        flat_match_counts = np.bincount(
            alpha_indices * num_id_pairs + pair_indices[match_indices],
            minlength=num_alphas * num_id_pairs,
        )
        nonzero_match_indices = np.flatnonzero(flat_match_counts)
        match_alpha_indices = nonzero_match_indices // num_id_pairs
        matched_pair_indices = nonzero_match_indices % num_id_pairs
        match_gt_indices = matched_pair_indices // num_tracker_ids
        match_tracker_indices = matched_pair_indices % num_tracker_ids
        match_counts = flat_match_counts[nonzero_match_indices].astype(float, copy=False)

    false_negatives = num_objects - true_positives
    false_positives = num_predictions - true_positives
    detre = metrics_module._quiet_divide(
        true_positives,
        np.maximum(1, true_positives + false_negatives),
    )
    detpr = metrics_module._quiet_divide(
        true_positives,
        np.maximum(1, true_positives + false_positives),
    )
    deta = metrics_module._quiet_divide(
        true_positives,
        np.maximum(1, true_positives + false_negatives + false_positives),
    )
    assa, assre, asspr = _compute_hota_association_scores(
        match_alpha_indices,
        match_gt_indices,
        match_tracker_indices,
        match_counts,
        gt_id_counts,
        tracker_id_counts,
        true_positives,
    )
    loca = np.maximum(1e-10, localization_sums) / np.maximum(1e-10, true_positives)
    hota = np.sqrt(deta * assa)
    owta = np.sqrt(detre * assa)
    return {
        "hota_alpha": hota,
        "deta_alpha": deta,
        "assa_alpha": assa,
        "detre_alpha": detre,
        "detpr_alpha": detpr,
        "assre_alpha": assre,
        "asspr_alpha": asspr,
        "loca_alpha": loca,
        "owta_alpha": owta,
        "num_detections": true_positives,
        "num_misses": false_negatives,
        "num_objects": num_objects,
        "num_false_positives": false_positives,
    }


def _prepare_iou_sequence_data(
    gt,
    test,
    distth,
    distfields=None,
    progress=None,
    event_recorder=None,
):
    if distfields is None:
        distfields = ["X", "Y", "Width", "Height"]

    gt_ids = np.unique(gt.ids)
    tracker_ids = np.unique(test.ids)
    gt_groups, gt_id_counts = _group_frame_arrays(gt, gt_ids, distfields)
    test_groups, tracker_id_counts = _group_frame_arrays(test, tracker_ids, distfields)
    frame_ids = np.union1d(gt.frame_ids, test.frame_ids)

    potential_matches = np.zeros((len(gt_ids), len(tracker_ids)))
    accumulator = _Accumulator(
        gt_id_counts,
        tracker_id_counts,
        event_recorder=event_recorder,
    )
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
        frame_data.append((frame_gt_indices, frame_tracker_indices, similarities))
        distances = 1 - similarities
        finite = distances <= distth
        accumulator.update(
            frame_gt_indices,
            frame_tracker_indices,
            distances,
            finite,
            frame_id=frame_id,
        )
        if similarities.size == 0:
            continue

        similarity_denominator = similarities.sum(0)[np.newaxis, :] + similarities.sum(1)[:, np.newaxis] - similarities
        similarity_iou = np.zeros_like(similarities)
        similarity_mask = similarity_denominator > _FLOAT_EPS
        similarity_iou[similarity_mask] = similarities[similarity_mask] / similarity_denominator[similarity_mask]
        potential_matches[
            frame_gt_indices[:, None],
            frame_tracker_indices,
        ] += similarity_iou

    alignment_scores = metrics_module._quiet_divide(
        potential_matches,
        np.maximum(1, gt_id_counts[:, np.newaxis] + tracker_id_counts[np.newaxis, :] - potential_matches),
    )
    return _PreparedIoUSequence(
        frame_data=frame_data,
        gt_id_counts=gt_id_counts,
        tracker_id_counts=tracker_id_counts,
        alignment_scores=alignment_scores,
        frame_ids=frame_ids,
        ground_truth_ids=gt_ids,
        tracker_ids=tracker_ids,
        num_objects=len(gt),
        num_predictions=len(test),
        accumulator=accumulator,
    )


def _group_frame_arrays(data, unique_ids, fields):
    """Group compact MOT columns into per-frame array views."""
    frame_ids = data.frame_ids
    id_codes = np.searchsorted(unique_ids, data.ids)
    values = data.values(fields)
    if len(frame_ids) == 0:
        return {}, np.zeros(len(unique_ids))

    if np.any(frame_ids[1:] < frame_ids[:-1]):
        order = np.argsort(frame_ids, kind="stable")
        frame_ids = frame_ids[order]
        id_codes = id_codes[order]
        values = values[order]

    boundaries = np.flatnonzero(frame_ids[1:] != frame_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(frame_ids)]))
    groups = {}
    id_counts = np.bincount(id_codes, minlength=len(unique_ids)).astype(float, copy=False)
    for start, end in zip(starts, ends):
        frame_id_codes = id_codes[start:end]
        groups[frame_ids[start]] = (frame_id_codes, values[start:end])
    return groups, id_counts


def _compute_hota_association_scores(
    alpha_indices,
    gt_indices,
    tracker_indices,
    match_counts,
    gt_id_counts,
    tracker_id_counts,
    true_positives,
):
    if len(match_counts) == 0:
        zeros = np.zeros_like(true_positives)
        return zeros, zeros.copy(), zeros.copy()
    true_positive_denominator = np.maximum(1, true_positives)
    squared_match_counts = match_counts * match_counts

    def aggregate(denominator):
        weighted_counts = squared_match_counts / np.maximum(1, denominator)
        return np.bincount(
            alpha_indices,
            weights=weighted_counts,
            minlength=len(true_positives),
        ) / true_positive_denominator

    assa = aggregate(gt_id_counts[gt_indices] + tracker_id_counts[tracker_indices] - match_counts)
    assre = aggregate(gt_id_counts[gt_indices])
    asspr = aggregate(tracker_id_counts[tracker_indices])
    return assa, assre, asspr


def _combine_hota_sequence_summaries(summaries):
    summaries = list(summaries)
    true_positives = np.sum([summary["num_detections"] for summary in summaries], axis=0)
    false_negatives = np.sum([summary["num_misses"] for summary in summaries], axis=0)
    false_positives = np.sum([summary["num_false_positives"] for summary in summaries], axis=0)
    num_objects = sum(summary["num_objects"] for summary in summaries)
    detre = metrics_module._quiet_divide(
        true_positives,
        np.maximum(1, true_positives + false_negatives),
    )
    detpr = metrics_module._quiet_divide(
        true_positives,
        np.maximum(1, true_positives + false_positives),
    )
    deta = metrics_module._quiet_divide(
        true_positives,
        np.maximum(1, true_positives + false_negatives + false_positives),
    )
    weighted_denominator = np.maximum(1, true_positives)

    def combine_weighted(metric):
        weighted_sum = np.sum(
            [summary[metric] * summary["num_detections"] for summary in summaries],
            axis=0,
        )
        return metrics_module._quiet_divide(weighted_sum, weighted_denominator)

    assa = combine_weighted("assa_alpha")
    assre = combine_weighted("assre_alpha")
    asspr = combine_weighted("asspr_alpha")
    localization_sum = np.sum(
        [summary["loca_alpha"] * summary["num_detections"] for summary in summaries],
        axis=0,
    )
    loca = np.maximum(1e-10, localization_sum) / np.maximum(1e-10, true_positives)
    hota = np.sqrt(deta * assa)
    owta = np.sqrt(detre * assa)
    return {
        "hota_alpha": hota,
        "deta_alpha": deta,
        "assa_alpha": assa,
        "detre_alpha": detre,
        "detpr_alpha": detpr,
        "assre_alpha": assre,
        "asspr_alpha": asspr,
        "loca_alpha": loca,
        "owta_alpha": owta,
        "num_detections": true_positives,
        "num_misses": false_negatives,
        "num_objects": num_objects,
        "num_false_positives": false_positives,
    }


def _prepare_metrics(metric_names, exclude_id):
    if metric_names is None:
        metric_names = list(metrics_module._MOTCHALLENGE_METRICS)
    elif isinstance(metric_names, str):
        metric_names = [metric_names]
    else:
        metric_names = list(metric_names)

    if exclude_id:
        metric_names = [metric_name for metric_name in metric_names if not metric_name.startswith("id")]
    return metric_names


def _prepare_metric_families(families, n_jobs):
    """Validate explicit extensions before any worker processes are started."""
    families = _coerce_metric_families(families)
    if not families:
        return ()
    reserved_metric_names = set(metrics_module._METRIC_SPECS)
    reserved_metric_names.update(HOTA_SUMMARY_METRICS.values())
    family_names = set()
    for family in families:
        _validate_metric_family(
            family,
            family_names,
            reserved_metric_names,
        )

    if n_jobs > 1 and families:
        import pickle

        try:
            pickle.dumps(families, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            raise TypeError(
                "Metric family instances must be picklable when n_jobs > 1."
            ) from exc
    return families


def _coerce_metric_families(families):
    if families is None:
        return ()
    if isinstance(families, MetricFamily):
        return (families,)
    try:
        return tuple(families)
    except TypeError as exc:
        raise TypeError(
            "extra_metric_families must be a MetricFamily or an iterable "
            "of MetricFamily instances."
        ) from exc


def _validate_metric_family(family, family_names, reserved_metric_names):
    if not isinstance(family, MetricFamily):
        raise TypeError(
            "Every extra metric family must inherit motmetrics.MetricFamily."
        )
    if not isinstance(family.name, str) or not family.name:
        raise ValueError("Every metric family must declare a non-empty name.")
    if family.name in family_names:
        raise ValueError("Duplicate metric family name: {!r}.".format(family.name))
    family_names.add(family.name)
    if any(
        not callable(getattr(family, method_name, None))
        for method_name in ("evaluate_sequence", "summarize", "combine")
    ):
        raise TypeError(
            "Metric family {!r} must implement evaluate_sequence, summarize, "
            "and combine.".format(family.name)
        )

    family_metric_names = _validate_metric_family_names(family)
    collisions = reserved_metric_names.intersection(family_metric_names)
    if collisions:
        raise ValueError(
            "Metric family {!r} reuses existing metric names: {}.".format(
                family.name,
                ", ".join(sorted(collisions)),
            )
        )
    reserved_metric_names.update(family_metric_names)
    _validate_metric_family_requirements(family)
    _validate_metric_family_metadata(family, family_metric_names)


def _validate_metric_family_names(family):
    if (
        isinstance(family.metric_names, str)
        or not isinstance(family.metric_names, Sequence)
    ):
        raise TypeError(
            "Metric family {!r} metric_names must be an ordered sequence.".format(
                family.name
            )
        )
    metric_names = tuple(family.metric_names)
    if not metric_names:
        raise ValueError(
            "Metric family {!r} must declare at least one metric name.".format(
                family.name
            )
        )
    if any(not isinstance(name, str) or not name for name in metric_names):
        raise ValueError(
            "Metric family {!r} has an invalid metric name.".format(family.name)
        )
    if len(set(metric_names)) != len(metric_names):
        raise ValueError(
            "Metric family {!r} contains duplicate metric names.".format(family.name)
        )
    return metric_names


def _validate_metric_family_requirements(family):
    if (
        isinstance(family.requirements, str)
        or not isinstance(family.requirements, Collection)
    ):
        raise TypeError(
            "Metric family {!r} requirements must be a reusable collection "
            "of names.".format(family.name)
        )
    requirements = frozenset(family.requirements)
    unknown_requirements = requirements - SUPPORTED_INTERMEDIATES
    if unknown_requirements:
        raise ValueError(
            "Metric family {!r} requests unsupported intermediates: {}.".format(
                family.name,
                ", ".join(sorted(unknown_requirements)),
            )
        )


def _extend_rows_with_metric_families(
    rows,
    sequence_results,
    generate_overall,
    metric_families,
):
    for family_index, family in enumerate(metric_families):
        for row, result in zip(rows[:len(sequence_results)], sequence_results):
            row.update(result[3][family_index][1])
        if generate_overall:
            partials = [
                result[3][family_index][0]
                for result in sequence_results
            ]
            rows[-1].update(
                _validate_metric_family_values(
                    family,
                    family.combine(partials),
                    "OVERALL",
                )
            )


def _validate_metric_family_metadata(family, metric_names):
    for attribute in ("display_names", "formatters"):
        values = getattr(family, attribute)
        if not isinstance(values, Mapping):
            raise TypeError(
                "Metric family {!r} {} must be a mapping.".format(
                    family.name,
                    attribute,
                )
            )
        unknown_names = set(values) - set(metric_names)
        if unknown_names:
            raise ValueError(
                "Metric family {!r} {} contains unknown metrics: {}.".format(
                    family.name,
                    attribute,
                    ", ".join(sorted(unknown_names)),
                )
            )
    if any(not isinstance(value, str) or not value for value in family.display_names.values()):
        raise ValueError(
            "Metric family {!r} display names must be non-empty strings.".format(
                family.name
            )
        )
    if any(not callable(value) for value in family.formatters.values()):
        raise TypeError(
            "Metric family {!r} formatters must be callable.".format(family.name)
        )


def _validate_metric_family_values(family, values, row_name):
    if not isinstance(values, Mapping):
        raise TypeError(
            "Metric family {!r} summarize/combine must return a mapping for {}.".format(
                family.name,
                row_name,
            )
        )
    metric_names = tuple(family.metric_names)
    missing = set(metric_names) - set(values)
    unexpected = set(values) - set(metric_names)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing {}".format(", ".join(sorted(missing))))
        if unexpected:
            details.append("unexpected {}".format(", ".join(sorted(unexpected))))
        raise ValueError(
            "Metric family {!r} returned invalid metrics for {}: {}.".format(
                family.name,
                row_name,
                "; ".join(details),
            )
        )
    if any(not isinstance(values[name], Number) for name in metric_names):
        raise TypeError(
            "Metric family {!r} must return numeric scalar values for {}.".format(
                family.name,
                row_name,
            )
        )
    return OrderedDict((name, values[name]) for name in metric_names)


def _summary_formatters(metric_families=()):
    formatters = dict(metrics_module._FORMATTERS)
    formatters.update({metric: "{:.1%}".format for metric in HOTA_SUMMARY_METRICS.values()})
    for family in metric_families:
        formatters.update(family.formatters)
    return formatters


def _summary_namemap(metric_families=()):
    namemap = dict(io.motchallenge_metric_names)
    namemap.update({
        "hota": "HOTA",
        "deta": "DetA",
        "assa": "AssA",
        "detre": "DetRe",
        "detpr": "DetPr",
        "assre": "AssRe",
        "asspr": "AssPr",
        "loca": "LocA",
        "owta": "OWTA",
    })
    for family in metric_families:
        namemap.update(family.display_names)
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
