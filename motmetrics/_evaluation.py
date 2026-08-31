# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre, Mikel Broström and others.
# See LICENSE file for terms.

"""High-level helpers for common tracker evaluation workflows."""

import os
import sys
from collections import OrderedDict
from collections.abc import Collection, Mapping, Sequence
from numbers import Integral, Number
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
_BOX_FIELDS = ("X", "Y", "Width", "Height")
_BENCHMARK_CONFIG_DIR = Path(__file__).with_name("configs")
_BENCHMARK_CONFIG_CACHE = {}
_BENCHMARK_NAMES = None
_DISTRACTOR_MODES = frozenset(("iou_assignment", "prediction_coverage"))
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
    n_jobs=None,
    progress=None,
    extra_metric_families=None,
    benchmark=None,
    target_classes=None,
    distractor_classes=None,
    distractor_iou_threshold=None,
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
    metrics : str or iterable of str, optional
        Built-in CLEAR, Identity, and HOTA result columns to return. Defaults
        to every standard metric. Explicit selections preserve their order and
        skip HOTA computation when no HOTA metric is requested.
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
    benchmark : {"MOT15", "MOT16", "MOT17", "MOT20", "SPORTSMOT", "VISDRONE"}, optional
        Benchmark preprocessing profile. By default it is inferred from the
        input path or labeled ground truth. Profiles define target classes,
        distractors, class-aware matching, and ignore-region handling.
    target_classes : int or iterable of int, optional
        Ground-truth classes to score. Overrides the benchmark default.
    distractor_classes : int or iterable of int, optional
        Ground-truth classes whose matched predictions are ignored. Overrides
        the benchmark default; pass an empty iterable to disable suppression.
    distractor_iou_threshold : float, optional
        Minimum overlap used for distractor preprocessing. This is IoU for
        assignment profiles and prediction coverage for ignore-region profiles.
        Overrides the benchmark profile; defaults to 0.5 without a profile.

    Returns
    -------
    _MOTChallengeSummary
        Native MOTChallenge-style results. ``print(summary)`` does not import
        pandas; ``summary.df`` materializes a dataframe on demand.
    """
    gt_path = Path(groundtruths)
    test_path = Path(tests)
    _validate_paths(gt_path, test_path)
    matched_files = _match_input_files(gt_path, test_path, name)
    if not matched_files:
        raise ValueError("No matching ground-truth and tracker result files found.")

    n_jobs = _validate_n_jobs(n_jobs, num_tasks=len(matched_files))
    benchmark = _normalize_benchmark(benchmark)
    target_classes = _normalize_class_ids(target_classes, "target_classes")
    distractor_classes = _normalize_class_ids(
        distractor_classes,
        "distractor_classes",
        allow_empty=True,
    )
    if distractor_iou_threshold is not None:
        distractor_iou_threshold = _normalize_iou_threshold(
            distractor_iou_threshold
        )
    progress_enabled = _progress_is_enabled(progress)

    metric_names, core_metric_names, hota_metric_names = _prepare_metrics(
        metrics,
        exclude_id,
    )
    metric_families = _prepare_metric_families(
        extra_metric_families,
        n_jobs,
    )
    if hota_alphas is None:
        hota_alphas = HOTA_ALPHAS
    summary = _evaluate_iou_paths(
        matched_files,
        fmt,
        gt_min_confidence,
        distfields,
        distth,
        metric_names,
        core_metric_names,
        hota_metric_names,
        hota_alphas,
        generate_overall and not gt_path.is_file(),
        n_jobs,
        progress_enabled,
        metric_families,
        benchmark,
        target_classes,
        distractor_classes,
        distractor_iou_threshold,
    )

    return _MOTChallengeSummary(
        *summary,
        formatters=_summary_formatters(metric_families),
        namemap=_summary_namemap(metric_families),
    )


def _match_input_files(gt_root, test_root, sequence_name=None):
    if gt_root.is_file():
        return [(sequence_name or _default_sequence_name(test_root), gt_root, test_root)]
    gt_files = _find_groundtruth_files(gt_root)
    test_files = _find_test_files(test_root)
    return [
        (name, gt_files[name], test_path)
        for name, test_path in test_files.items()
        if name in gt_files
    ]


def _evaluate_iou_paths(
    matched_files,
    fmt,
    gt_min_confidence,
    distfields,
    distth,
    metric_names,
    core_metric_names,
    hota_metric_names,
    hota_alphas,
    generate_overall,
    n_jobs,
    progress,
    metric_families,
    benchmark,
    target_classes,
    distractor_classes,
    distractor_iou_threshold,
):
    """Evaluate every input through the canonical state-only IoU engine."""
    tasks = [
        (
            task_index,
            name,
            gt_path,
            test_path,
            fmt,
            gt_min_confidence,
            benchmark,
            target_classes,
            distractor_classes,
            distractor_iou_threshold,
            distfields,
            distth,
            core_metric_names,
            hota_metric_names,
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
    rows = [
        OrderedDict(
            (metric, partial[metric])
            for metric in core_metric_names
        )
        for partial in partials
    ]
    result_names = list(names)
    if generate_overall:
        rows.append(
            metrics_module._compute_overall(
                partials,
                metric_names=core_metric_names,
            )
        )
        result_names.append("OVERALL")
    _extend_rows_with_hota(
        rows,
        result_names,
        results,
        hota_metric_names,
        generate_overall,
    )
    _extend_rows_with_metric_families(
        rows,
        results,
        generate_overall,
        metric_families,
    )
    columns = list(metric_names)
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
        benchmark,
        target_classes,
        distractor_classes,
        distractor_iou_threshold,
        distfields,
        distth,
        core_metric_names,
        hota_metric_names,
        hota_alphas,
        metric_families,
    ) = task
    sequence_progress = _WorkerSequenceProgress(task_index) if _WORKER_PROGRESS_STAGE is not None else None
    if sequence_progress is not None:
        sequence_progress.stage(_PROGRESS_LOADING)
    try:
        ground_truth = io.loadtxt(gt_path, fmt=fmt, min_confidence=-np.inf)
        tracker = io.loadtxt(test_path, fmt=fmt)
        ground_truth, tracker, preprocessing_ious = _preprocess_motchallenge_inputs(
            ground_truth,
            tracker,
            gt_path,
            benchmark,
            gt_min_confidence,
            target_classes,
            distractor_classes,
            distractor_iou_threshold,
        )
        event_recorder = _create_clear_event_recorder(
            metric_families,
            ground_truth,
            tracker,
        )
        compute_hota = bool(hota_metric_names)
        retain_frame_iou = compute_hota or any(
            "frame_iou" in family.requirements
            for family in metric_families
        )
        prepared = _prepare_iou_sequence_data(
            ground_truth,
            tracker,
            distth,
            distfields,
            progress=sequence_progress,
            event_recorder=event_recorder,
            compute_hota=compute_hota,
            retain_frame_iou=retain_frame_iou,
            precomputed_similarities=(
                preprocessing_ious
                if _uses_box_fields(distfields)
                else None
            ),
        )

        if sequence_progress is not None:
            sequence_progress.stage(_PROGRESS_METRICS)
        partial = metrics_module._compute_metrics(
            prepared.accumulator,
            metric_names=core_metric_names,
        )
        hota_summary = None
        if compute_hota:
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

    def begin(self, frame_count, include_hota=True):
        """Set the frame-derived total and enter the IoU pass."""
        self.total = frame_count * (2 if include_hota else 1)
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


def _preprocess_motchallenge_inputs(
    ground_truth,
    tracker,
    ground_truth_path,
    benchmark,
    gt_min_confidence,
    target_classes,
    distractor_classes,
    distractor_iou_threshold,
):
    """Apply benchmark defaults or caller-supplied class preprocessing."""
    confidence_keep = _ground_truth_confidence_mask(
        ground_truth,
        gt_min_confidence,
    )
    resolved_benchmark = _resolve_motchallenge_benchmark(
        benchmark,
        ground_truth_path,
        ground_truth,
    )
    protocol = _resolve_class_protocol(
        resolved_benchmark,
        target_classes,
        distractor_classes,
        distractor_iou_threshold,
    )
    if protocol is None:
        return ground_truth._take(confidence_keep), tracker, None
    target_classes = protocol["target_classes"]
    distractor_classes = protocol["distractor_classes"]
    distractor_iou_threshold = protocol["distractor_threshold"]

    ground_truth_classes = _numeric_class_ids(ground_truth)
    if ground_truth_classes is None:
        raise ValueError(
            "Class preprocessing requires integer ground-truth ClassId values."
        )
    tracker_classes = _numeric_class_ids(tracker)
    _validate_preprocessing_classes(
        protocol,
        resolved_benchmark,
        ground_truth_classes,
        tracker_classes,
    )

    ground_truth_keep = confidence_keep & np.isin(
        ground_truth_classes,
        target_classes,
    )
    tracker_keep = np.ones(len(tracker), dtype=np.bool_)
    if protocol["filter_tracker_classes"]:
        tracker_keep &= np.isin(tracker_classes, target_classes)

    ground_truth_rows = _group_frame_row_indices(ground_truth.frame_ids)
    tracker_rows = _group_frame_row_indices(tracker.frame_ids)
    ground_truth_boxes = ground_truth.values(_BOX_FIELDS)
    tracker_boxes = tracker.values(_BOX_FIELDS)
    retained_ious = {}
    empty_rows = np.empty(0, dtype=np.intp)
    for frame_id in ground_truth_rows.keys() | tracker_rows.keys():
        frame_ground_truth_rows = ground_truth_rows.get(frame_id, empty_rows)
        frame_tracker_rows = tracker_rows.get(frame_id, empty_rows)
        frame_classes = ground_truth_classes[frame_ground_truth_rows]
        frame_ground_truth_keep = ground_truth_keep[frame_ground_truth_rows]
        frame_tracker_keep = tracker_keep[frame_tracker_rows].copy()
        frame_distractor_rows = np.isin(
            frame_classes,
            distractor_classes,
        )
        if np.any(frame_distractor_rows):
            if protocol["distractor_mode"] == "iou_assignment":
                _suppress_iou_assignment_matches(
                    ground_truth_boxes[frame_ground_truth_rows],
                    frame_classes,
                    tracker_boxes[frame_tracker_rows],
                    frame_tracker_keep,
                    distractor_classes,
                    distractor_iou_threshold,
                )
                tracker_keep[frame_tracker_rows] = frame_tracker_keep
            else:
                _suppress_prediction_coverage_matches(
                    ground_truth_boxes,
                    tracker_boxes,
                    frame_ground_truth_rows,
                    frame_tracker_rows,
                    frame_distractor_rows,
                    frame_ground_truth_keep,
                    frame_tracker_keep,
                    ground_truth_keep,
                    tracker_keep,
                    distractor_iou_threshold,
                    protocol["suppress_target_ground_truth"],
                )

        frame_ground_truth_rows = frame_ground_truth_rows[
            frame_ground_truth_keep
        ]
        frame_tracker_rows = frame_tracker_rows[frame_tracker_keep]
        similarities = iou_matrix(
            ground_truth_boxes[frame_ground_truth_rows],
            tracker_boxes[frame_tracker_rows],
            return_dist=False,
        )
        if protocol["class_aware"] and similarities.size:
            similarities[
                ground_truth_classes[frame_ground_truth_rows, None]
                != tracker_classes[frame_tracker_rows]
            ] = 0
        retained_ious[frame_id] = similarities

    ground_truth = ground_truth._take(ground_truth_keep)
    tracker = tracker._take(tracker_keep)
    if protocol["class_aware"]:
        ground_truth = _with_class_aware_ids(ground_truth)
        tracker = _with_class_aware_ids(tracker)
    return ground_truth, tracker, retained_ious


def _validate_preprocessing_classes(
    protocol,
    benchmark,
    ground_truth_classes,
    tracker_classes,
):
    valid_classes = protocol["valid_classes"]
    if valid_classes is not None:
        invalid_classes = np.setdiff1d(
            np.unique(ground_truth_classes),
            valid_classes,
        )
        if len(invalid_classes):
            raise ValueError(
                "Invalid {} ground-truth class IDs: {}".format(
                    benchmark,
                    ", ".join(str(value) for value in invalid_classes),
                )
            )

    requires_tracker_classes = (
        protocol["valid_tracker_classes"] is not None
        or protocol["filter_tracker_classes"]
        or protocol["class_aware"]
    )
    if requires_tracker_classes and tracker_classes is None:
        raise ValueError(
            "{} preprocessing requires integer tracker ClassId values.".format(
                benchmark,
            )
        )
    valid_tracker_classes = protocol["valid_tracker_classes"]
    if valid_tracker_classes is not None and len(tracker_classes):
        invalid_tracker_classes = np.setdiff1d(
            np.unique(tracker_classes),
            valid_tracker_classes,
        )
        if len(invalid_tracker_classes):
            raise ValueError(
                "Invalid {} tracker class IDs: {}".format(
                    benchmark,
                    ", ".join(str(value) for value in invalid_tracker_classes),
                )
            )
    tracker_max_class = protocol["tracker_max_class"]
    if (
        tracker_max_class is not None
        and tracker_classes is not None
        and len(tracker_classes)
        and np.max(tracker_classes) > tracker_max_class
    ):
        raise ValueError(
            "{} evaluation does not accept tracker class IDs greater than {}.".format(
                benchmark,
                tracker_max_class,
            )
        )


def _suppress_prediction_coverage_matches(
    ground_truth_boxes,
    tracker_boxes,
    frame_ground_truth_rows,
    frame_tracker_rows,
    frame_distractor_rows,
    frame_ground_truth_keep,
    frame_tracker_keep,
    ground_truth_keep,
    tracker_keep,
    threshold,
    suppress_target_ground_truth,
):
    distractor_boxes = ground_truth_boxes[
        frame_ground_truth_rows[frame_distractor_rows]
    ]
    if suppress_target_ground_truth:
        target_positions = np.flatnonzero(frame_ground_truth_keep)
        target_suppressed = _covered_by_regions(
            ground_truth_boxes[frame_ground_truth_rows[target_positions]],
            distractor_boxes,
            threshold,
        )
        suppressed_positions = target_positions[target_suppressed]
        frame_ground_truth_keep[suppressed_positions] = False
        ground_truth_keep[frame_ground_truth_rows[suppressed_positions]] = False

    tracker_positions = np.flatnonzero(frame_tracker_keep)
    tracker_suppressed = _covered_by_regions(
        tracker_boxes[frame_tracker_rows[tracker_positions]],
        distractor_boxes,
        threshold,
    )
    suppressed_positions = tracker_positions[tracker_suppressed]
    frame_tracker_keep[suppressed_positions] = False
    tracker_keep[frame_tracker_rows[suppressed_positions]] = False


def _suppress_iou_assignment_matches(
    ground_truth_boxes,
    ground_truth_classes,
    tracker_boxes,
    tracker_keep,
    distractor_classes,
    threshold,
):
    tracker_positions = np.flatnonzero(tracker_keep)
    if not len(tracker_positions):
        return
    distractor_rows = np.isin(ground_truth_classes, distractor_classes)
    distractor_similarities = iou_matrix(
        ground_truth_boxes[distractor_rows],
        tracker_boxes[tracker_positions],
        return_dist=False,
    )
    if not np.any(distractor_similarities >= threshold - _FLOAT_EPS):
        return

    matching_scores = iou_matrix(
        ground_truth_boxes,
        tracker_boxes[tracker_positions],
        return_dist=False,
    )
    matching_scores[matching_scores < threshold - _FLOAT_EPS] = 0
    matched_rows, matched_columns = _dense_linear_sum_assignment(
        -matching_scores
    )
    actually_matched = matching_scores[matched_rows, matched_columns] > _FLOAT_EPS
    matched_rows = matched_rows[actually_matched]
    matched_columns = matched_columns[actually_matched]
    distractor_matches = np.isin(
        ground_truth_classes[matched_rows],
        distractor_classes,
    )
    tracker_keep[tracker_positions[matched_columns[distractor_matches]]] = False


def _covered_by_regions(boxes, regions, threshold):
    """Return boxes whose area is covered by the union of ignore regions."""
    covered = np.zeros(len(boxes), dtype=np.bool_)
    for index, (x, y, width, height) in enumerate(boxes):
        area = width * height
        if area <= _FLOAT_EPS:
            continue
        left = np.maximum(x, regions[:, 0])
        top = np.maximum(y, regions[:, 1])
        right = np.minimum(x + width, regions[:, 0] + regions[:, 2])
        bottom = np.minimum(y + height, regions[:, 1] + regions[:, 3])
        intersections = np.column_stack((left, top, right, bottom))
        intersections = intersections[
            (right > left) & (bottom > top)
        ]
        if len(intersections):
            covered[index] = (
                _rectangle_union_area(intersections)
                >= threshold * area - _FLOAT_EPS
            )
    return covered


def _rectangle_union_area(rectangles):
    x_coordinates = np.unique(rectangles[:, (0, 2)])
    area = 0.0
    for left, right in zip(x_coordinates[:-1], x_coordinates[1:]):
        if right <= left:
            continue
        active = rectangles[
            (rectangles[:, 0] < right) & (rectangles[:, 2] > left)
        ]
        intervals = active[np.argsort(active[:, 1]), 1:4:2]
        covered_height = 0.0
        start = end = None
        for top, bottom in intervals:
            if start is None:
                start, end = top, bottom
            elif top > end:
                covered_height += end - start
                start, end = top, bottom
            else:
                end = max(end, bottom)
        if start is not None:
            covered_height += end - start
        area += (right - left) * covered_height
    return area


def _with_class_aware_ids(data):
    classes = _numeric_class_ids(data)
    pairs = np.column_stack((classes, data.ids))
    _, ids = np.unique(pairs, axis=0, return_inverse=True)
    return data._with_ids(ids)


def _ground_truth_confidence_mask(ground_truth, min_confidence):
    if "Confidence" not in ground_truth.field_names:
        return np.ones(len(ground_truth), dtype=np.bool_)
    return ground_truth.column("Confidence") >= min_confidence


def _resolve_motchallenge_benchmark(benchmark, ground_truth_path, ground_truth):
    if benchmark is not None:
        return benchmark

    upper_path = str(ground_truth_path).upper()
    for candidate in _benchmark_names():
        if candidate in upper_path:
            return candidate

    if _looks_like_labeled_motchallenge(ground_truth):
        return "MOT17"
    return None


def _resolve_class_protocol(
    benchmark,
    target_classes,
    distractor_classes,
    distractor_iou_threshold,
):
    config = _load_benchmark_config(benchmark) if benchmark is not None else None
    default_target_classes = config["target_classes"] if config is not None else None
    default_distractor_classes = config["distractor_classes"] if config is not None else None
    if (
        default_target_classes is None
        and target_classes is None
        and distractor_classes is None
    ):
        return None

    if target_classes is None:
        target_classes = default_target_classes or (1,)
    if distractor_classes is None:
        distractor_classes = default_distractor_classes or ()
    if distractor_iou_threshold is None:
        distractor_iou_threshold = (
            config["distractor_threshold"]
            if config is not None
            else 0.5
        )
    overlap = set(target_classes) & set(distractor_classes)
    if overlap:
        raise ValueError(
            "target_classes and distractor_classes must not overlap: {}".format(
                ", ".join(str(value) for value in sorted(overlap))
            )
        )
    uses_benchmark_classes = (
        config is not None
        and target_classes == default_target_classes
        and distractor_classes == default_distractor_classes
    )
    return {
        "target_classes": target_classes,
        "distractor_classes": distractor_classes,
        "distractor_threshold": distractor_iou_threshold,
        "valid_classes": (
            config["valid_classes"] if uses_benchmark_classes else None
        ),
        "tracker_max_class": (
            config["tracker_max_class"] if uses_benchmark_classes else None
        ),
        "valid_tracker_classes": (
            config["valid_tracker_classes"] if uses_benchmark_classes else None
        ),
        "class_aware": config["class_aware"] if config is not None else False,
        "filter_tracker_classes": (
            config["filter_tracker_classes"] if config is not None else False
        ),
        "distractor_mode": (
            config["distractor_mode"]
            if config is not None
            else "iou_assignment"
        ),
        "suppress_target_ground_truth": (
            config["suppress_target_ground_truth"]
            if config is not None
            else False
        ),
    }


def _looks_like_labeled_motchallenge(ground_truth):
    classes = _numeric_class_ids(ground_truth)
    if classes is None or not len(classes):
        return False
    valid_classes = _load_benchmark_config("MOT17")["valid_classes"]
    if valid_classes is None or not np.all(np.isin(classes, valid_classes)):
        return False
    if "Confidence" not in ground_truth.field_names:
        return False
    marks = ground_truth.column("Confidence")
    if not np.all((marks == 0) | (marks == 1)):
        return False
    if "Visibility" not in ground_truth.field_names:
        return False
    visibility = ground_truth.column("Visibility")
    return np.all((visibility >= 0) & (visibility <= 1))


def _numeric_class_ids(data):
    if "ClassId" not in data.field_names:
        return None
    try:
        values = np.asarray(data.column("ClassId"), dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
        return None
    return values.astype(np.int64, copy=False)


def _group_frame_row_indices(frame_ids):
    if len(frame_ids) == 0:
        return {}
    if np.any(frame_ids[1:] < frame_ids[:-1]):
        order = np.argsort(frame_ids, kind="stable")
        sorted_frame_ids = frame_ids[order]
    else:
        order = np.arange(len(frame_ids), dtype=np.intp)
        sorted_frame_ids = frame_ids
    boundaries = np.flatnonzero(sorted_frame_ids[1:] != sorted_frame_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(sorted_frame_ids)]))
    return {
        sorted_frame_ids[start]: order[start:end]
        for start, end in zip(starts, ends)
    }


def _uses_box_fields(distfields):
    return distfields is None or tuple(distfields) == _BOX_FIELDS


def _prepare_iou_sequence_data(
    gt,
    test,
    distth,
    distfields=None,
    progress=None,
    event_recorder=None,
    compute_hota=True,
    retain_frame_iou=True,
    precomputed_similarities=None,
):
    if distfields is None:
        distfields = list(_BOX_FIELDS)

    gt_ids = np.unique(gt.ids)
    tracker_ids = np.unique(test.ids)
    gt_groups, gt_id_counts = _group_frame_arrays(gt, gt_ids, distfields)
    test_groups, tracker_id_counts = _group_frame_arrays(test, tracker_ids, distfields)
    frame_ids = np.union1d(gt.frame_ids, test.frame_ids)

    potential_matches = None
    if compute_hota:
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
        progress.begin(len(frame_ids), include_hota=compute_hota)

    for frame_id in frame_ids:
        if progress is not None:
            progress.advance()
        frame_gt_indices, frame_gt_values = gt_groups.get(frame_id, (empty_indices, empty_values))
        frame_tracker_indices, frame_tracker_values = test_groups.get(frame_id, (empty_indices, empty_values))
        if precomputed_similarities is not None and frame_id in precomputed_similarities:
            similarities = precomputed_similarities[frame_id]
        else:
            similarities = iou_matrix(frame_gt_values, frame_tracker_values, return_dist=False)
        if retain_frame_iou:
            frame_data.append((
                frame_gt_indices,
                frame_tracker_indices,
                similarities,
            ))
        distances = 1 - similarities
        finite = distances <= distth
        accumulator.update(
            frame_gt_indices,
            frame_tracker_indices,
            distances,
            finite,
            frame_id=frame_id,
        )
        if not compute_hota or similarities.size == 0:
            continue

        similarity_denominator = similarities.sum(0)[np.newaxis, :] + similarities.sum(1)[:, np.newaxis] - similarities
        similarity_iou = np.zeros_like(similarities)
        similarity_mask = similarity_denominator > _FLOAT_EPS
        similarity_iou[similarity_mask] = similarities[similarity_mask] / similarity_denominator[similarity_mask]
        potential_matches[
            frame_gt_indices[:, None],
            frame_tracker_indices,
        ] += similarity_iou

    alignment_scores = None
    if compute_hota:
        alignment_scores = metrics_module._quiet_divide(
            potential_matches,
            np.maximum(
                1,
                gt_id_counts[:, np.newaxis]
                + tracker_id_counts[np.newaxis, :]
                - potential_matches,
            ),
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
        metric_names = (
            list(metrics_module._MOTCHALLENGE_METRICS)
            + list(HOTA_SUMMARY_METRICS.values())
        )
    elif isinstance(metric_names, str):
        metric_names = [metric_names]
    else:
        metric_names = list(metric_names)

    if any(not isinstance(metric_name, str) for metric_name in metric_names):
        raise TypeError("metrics must contain only metric-name strings.")
    if len(set(metric_names)) != len(metric_names):
        raise ValueError("metrics must not contain duplicate names.")
    supported_names = set(metrics_module._METRIC_SPECS)
    supported_names.update(HOTA_SUMMARY_METRICS.values())
    unknown_names = set(metric_names) - supported_names
    if unknown_names:
        raise ValueError(
            "Unknown metric: {}".format(", ".join(sorted(unknown_names)))
        )
    if exclude_id:
        metric_names = [
            metric_name
            for metric_name in metric_names
            if not metric_name.startswith("id")
        ]
    core_metric_names = [
        metric_name
        for metric_name in metric_names
        if metric_name in metrics_module._METRIC_SPECS
    ]
    hota_metric_names = [
        metric_name
        for metric_name in metric_names
        if metric_name in HOTA_SUMMARY_METRICS.values()
    ]
    return metric_names, core_metric_names, hota_metric_names


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


def _extend_rows_with_hota(
    rows,
    result_names,
    sequence_results,
    hota_metric_names,
    generate_overall,
):
    if not hota_metric_names:
        return
    summaries = OrderedDict(
        (result[0], result[2])
        for result in sequence_results
    )
    if generate_overall:
        summaries["OVERALL"] = _combine_hota_sequence_summaries(
            summaries.values()
        )
    alpha_names = {
        summary_name: alpha_name
        for alpha_name, summary_name in HOTA_SUMMARY_METRICS.items()
    }
    for row_name, row in zip(result_names, rows):
        row.update(
            (
                metric_name,
                np.mean(summaries[row_name][alpha_names[metric_name]]),
            )
            for metric_name in hota_metric_names
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


def _validate_n_jobs(n_jobs, num_tasks=1, is_folder=False):
    # Auto-scaling worker count when n_jobs is None:
    # n_jobs = min(num_tasks, max(1, cpu_count - 2))
    #
    # Examples across datasets (assuming C=14 system CPUs):
    # 1. 2 sequences (e.g. TUD-Campus & TUD-Stadtmitte): N=2 -> min(2, 12) = 2 workers (no idle processes).
    # 2. 4 sequences (e.g. MOT20): N=4 -> min(4, 12) = 4 workers.
    # 3. 7 sequences (e.g. MOT17): N=7 -> min(7, 12) = 7 workers.
    # 4. 20 sequences: N=20 -> min(20, 12) = 12 workers (leaving 2 CPUs free for OS tasks).
    # 5. Single sequence file: N=1 -> 1 worker (executes in main process without IPC/spawn overhead).
    # 6. Explicit n_jobs=N or --jobs N overrides auto-scaling.
    if n_jobs is None:
        if num_tasks <= 1 and not is_folder:
            return 1
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
        max_cpus = max(1, cpu_count - 2)
        if num_tasks > 1:
            return min(num_tasks, max_cpus)
        return max_cpus
    if not isinstance(n_jobs, (int, np.integer)):
        raise TypeError("n_jobs must be an integer or None.")
    n_jobs = int(n_jobs)
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1.")
    return n_jobs


def _normalize_benchmark(benchmark):
    if benchmark is None:
        return None
    if not isinstance(benchmark, str):
        raise TypeError("benchmark must be a MOTChallenge benchmark name or None.")
    benchmark = benchmark.upper()
    benchmark_names = _benchmark_names()
    if benchmark not in benchmark_names:
        raise ValueError(
            "benchmark must be one of {}.".format(
                _format_choices(benchmark_names)
            )
        )
    return benchmark


def _benchmark_names():
    global _BENCHMARK_NAMES
    if _BENCHMARK_NAMES is None:
        _BENCHMARK_NAMES = tuple(
            path.stem.upper()
            for path in sorted(_BENCHMARK_CONFIG_DIR.glob("*.yaml"))
        )
    return _BENCHMARK_NAMES


def _load_benchmark_config(benchmark):
    """Load and validate one bundled JSON-compatible YAML profile."""
    cached = _BENCHMARK_CONFIG_CACHE.get(benchmark)
    if cached is not None:
        return cached

    import json

    path = _BENCHMARK_CONFIG_DIR / "{}.yaml".format(benchmark.lower())
    try:
        with path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "Could not load benchmark profile {}: {}".format(path, error)
        ) from error

    expected_fields = {
        "class_aware",
        "target_classes",
        "distractor_classes",
        "distractor_threshold",
        "distractor_mode",
        "filter_tracker_classes",
        "suppress_target_ground_truth",
        "valid_classes",
        "valid_tracker_classes",
        "tracker_max_class",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_fields:
        raise ValueError(
            "Benchmark profile {} must define exactly: {}.".format(
                path,
                ", ".join(sorted(expected_fields)),
            )
        )

    target_classes = _normalize_class_ids(
        raw["target_classes"],
        "target_classes",
    )
    distractor_classes = _normalize_class_ids(
        raw["distractor_classes"],
        "distractor_classes",
        allow_empty=True,
    )
    valid_classes = _normalize_class_ids(
        raw["valid_classes"],
        "valid_classes",
    )
    valid_tracker_classes = _normalize_class_ids(
        raw["valid_tracker_classes"],
        "valid_tracker_classes",
    )
    tracker_max_class = raw["tracker_max_class"]
    if tracker_max_class is not None and (
        not isinstance(tracker_max_class, Integral)
        or isinstance(tracker_max_class, (bool, np.bool_))
    ):
        raise TypeError("tracker_max_class must be an integer or null.")
    distractor_mode = raw["distractor_mode"]
    _validate_benchmark_config_values(
        raw,
        target_classes,
        distractor_classes,
        valid_classes,
        distractor_mode,
    )

    config = {
        "target_classes": target_classes,
        "distractor_classes": distractor_classes,
        "distractor_threshold": _normalize_iou_threshold(
            raw["distractor_threshold"]
        ),
        "valid_classes": valid_classes,
        "valid_tracker_classes": valid_tracker_classes,
        "tracker_max_class": (
            int(tracker_max_class) if tracker_max_class is not None else None
        ),
        "class_aware": raw["class_aware"],
        "filter_tracker_classes": raw["filter_tracker_classes"],
        "distractor_mode": distractor_mode,
        "suppress_target_ground_truth": raw["suppress_target_ground_truth"],
    }
    _BENCHMARK_CONFIG_CACHE[benchmark] = config
    return config


def _validate_benchmark_config_values(
    raw,
    target_classes,
    distractor_classes,
    valid_classes,
    distractor_mode,
):
    for field in (
        "class_aware",
        "filter_tracker_classes",
        "suppress_target_ground_truth",
    ):
        if not isinstance(raw[field], bool):
            raise TypeError("{} must be a boolean.".format(field))
    if distractor_mode not in _DISTRACTOR_MODES:
        raise ValueError(
            "distractor_mode must be one of {}.".format(
                _format_choices(tuple(sorted(_DISTRACTOR_MODES)))
            )
        )
    overlap = (
        set(target_classes) & set(distractor_classes)
        if target_classes is not None
        else set()
    )
    if overlap:
        raise ValueError(
            "Benchmark target and distractor classes overlap: {}.".format(
                ", ".join(str(value) for value in sorted(overlap))
            )
        )
    configured_classes = set(distractor_classes)
    if target_classes is not None:
        configured_classes.update(target_classes)
    unknown_classes = (
        configured_classes - set(valid_classes)
        if valid_classes is not None
        else set()
    )
    if unknown_classes:
        raise ValueError(
            "Benchmark classes are absent from valid_classes: {}.".format(
                ", ".join(str(value) for value in sorted(unknown_classes))
            )
        )


def _format_choices(values):
    if len(values) == 1:
        return values[0]
    return "{}, or {}".format(", ".join(values[:-1]), values[-1])


def _normalize_class_ids(values, name, allow_empty=False):
    if values is None:
        return None
    if isinstance(values, Integral) and not isinstance(values, (bool, np.bool_)):
        values = (int(values),)
    else:
        if isinstance(values, (str, bytes)):
            raise TypeError("{} must contain integer class IDs.".format(name))
        try:
            values = tuple(values)
        except TypeError:
            raise TypeError(
                "{} must be an integer or iterable of integers.".format(name)
            ) from None
        if any(
            not isinstance(value, Integral)
            or isinstance(value, (bool, np.bool_))
            for value in values
        ):
            raise TypeError("{} must contain integer class IDs.".format(name))
        values = tuple(int(value) for value in values)
    values = tuple(sorted(set(values)))
    if not values and not allow_empty:
        raise ValueError("{} must contain at least one class ID.".format(name))
    return values


def _normalize_iou_threshold(value):
    if not isinstance(value, Number) or isinstance(value, (bool, np.bool_)):
        raise TypeError("distractor_iou_threshold must be a number.")
    value = float(value)
    if not np.isfinite(value) or value < 0 or value > 1:
        raise ValueError("distractor_iou_threshold must be between 0 and 1.")
    return value


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
