# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""High-level helpers for common tracker evaluation workflows."""

from __future__ import absolute_import, division, print_function

import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from motmetrics import io, lap, utils
from motmetrics import metrics as metrics_module
from motmetrics.distances import iou_matrix
from motmetrics.mot import MOTAccumulator

HOTA_ALPHAS = np.arange(0.05, 0.99, 0.05)
HOTA_ALPHA_METRICS = ["hota_alpha", "deta_alpha", "assa_alpha"]
HOTA_SUMMARY_METRICS = OrderedDict([
    ("hota_alpha", "hota"),
    ("deta_alpha", "deta"),
    ("assa_alpha", "assa"),
])


class MOTChallengeSummary(object):
    """MOTChallenge metric results with both dataframe and rendered views."""

    def __init__(self, df, formatters=None, namemap=None):
        self.df = df
        self.formatters = formatters
        self.namemap = namemap

    @property
    def dataframe(self):
        """Alias for the raw pandas DataFrame."""
        return self.df

    @property
    def text(self):
        """Human-readable MOTChallenge-style table."""
        return io.render_summary(self.df, formatters=self.formatters, namemap=self.namemap)

    def render(self):
        """Return the human-readable MOTChallenge-style table."""
        return self.text

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        return iter(self.df)

    def __contains__(self, key):
        return key in self.df

    def __getitem__(self, key):
        return self.df[key]

    def __getattr__(self, name):
        return getattr(self.df, name)


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
    dist="iou",
    distfields=None,
    distth=0.5,
    metrics=None,
    name=None,
    generate_overall=True,
    gt_min_confidence=1,
    solver=None,
    id_solver=None,
    exclude_id=False,
    include_hota=True,
    hota_alphas=None,
    n_jobs=1,
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

    Returns
    -------
    MOTChallengeSummary
        Wrapper around the raw pandas DataFrame. ``print(summary)`` displays a
        MOTChallenge-style table, while ``summary.df`` exposes the dataframe.
    """
    gt_path = Path(groundtruths)
    test_path = Path(tests)
    _validate_paths(gt_path, test_path)
    _validate_n_jobs(n_jobs)

    if solver:
        lap.default_solver = solver

    metric_names = _prepare_metrics(metrics, exclude_id)
    metric_host = metrics_module.create()
    if hota_alphas is None:
        hota_alphas = HOTA_ALPHAS
    if include_hota and dist.upper() != "IOU":
        raise ValueError("HOTA is only supported with dist='iou'. Pass include_hota=False to skip HOTA metrics.")

    if gt_path.is_file() and test_path.is_file():
        sequence_name = name or _default_sequence_name(test_path)
        gt = io.loadtxt(gt_path, fmt=fmt, min_confidence=gt_min_confidence)
        test = io.loadtxt(test_path, fmt=fmt)
        acc, prepared = _compare_single_sequence(gt, test, dist, distfields, distth, include_hota)
        prepared_sequences = OrderedDict([(sequence_name, prepared)]) if prepared is not None else None

        if id_solver:
            lap.default_solver = id_solver
        summary = metric_host.compute(acc, metrics=metric_names, name=sequence_name)
        if include_hota:
            summary = _append_hota_summary(
                summary,
                OrderedDict([(sequence_name, gt)]),
                OrderedDict([(sequence_name, test)]),
                [sequence_name],
                metric_host,
                dist,
                distfields,
                hota_alphas,
                False,
                n_jobs,
                prepared_sequences=prepared_sequences,
            )
    else:
        gt = load_motchallenge_groundtruths(gt_path, fmt=fmt, min_confidence=gt_min_confidence)
        test = load_motchallenge_tests(test_path, fmt=fmt, names=gt)
        accs, names, prepared_sequences = _compare_multiple_sequences(
            gt,
            test,
            dist,
            distfields,
            distth,
            include_hota,
            n_jobs,
        )
        if not accs:
            raise ValueError("No matching ground-truth and tracker result files found.")

        if id_solver:
            lap.default_solver = id_solver
        summary = metric_host.compute_many(
            accs,
            names=names,
            metrics=metric_names,
            generate_overall=generate_overall,
            n_jobs=n_jobs,
        )
        if include_hota:
            summary = _append_hota_summary(
                summary,
                gt,
                test,
                names,
                metric_host,
                dist,
                distfields,
                hota_alphas,
                generate_overall,
                n_jobs,
                prepared_sequences=prepared_sequences,
            )

    return MOTChallengeSummary(
        summary,
        formatters=_summary_formatters(metric_host, include_hota),
        namemap=_summary_namemap(include_hota),
    )


def load_motchallenge_groundtruths(root, fmt=io.Format.AUTO, min_confidence=1):
    """Load MOTChallenge ground-truth dataframes from a file or folder."""
    root = Path(root)
    if root.is_file():
        return OrderedDict([(_default_sequence_name(root), io.loadtxt(root, fmt=fmt, min_confidence=min_confidence))])

    files = _find_groundtruth_files(root)
    if not files:
        raise ValueError("No ground-truth files found below {}.".format(root))
    logging.info("Found %d groundtruth files.", len(files))
    return OrderedDict((name, io.loadtxt(path, fmt=fmt, min_confidence=min_confidence)) for name, path in files.items())


def load_motchallenge_tests(root, fmt=io.Format.AUTO, names=None):
    """Load MOTChallenge tracker result dataframes from a file or folder."""
    root = Path(root)
    if root.is_file():
        return OrderedDict([(_default_sequence_name(root), io.loadtxt(root, fmt=fmt))])

    files = _find_test_files(root)
    if names is not None:
        files = OrderedDict((name, path) for name, path in files.items() if name in names)
    if not files:
        raise ValueError("No tracker result files found below {}.".format(root))
    logging.info("Found %d test files.", len(files))
    return OrderedDict((name, io.loadtxt(path, fmt=fmt)) for name, path in files.items())


def compare_dataframes(gts, tests, dist="iou", distfields=None, distth=0.5, n_jobs=1):
    """Build accumulators for matching ground-truth and tracker dataframes."""
    _validate_n_jobs(n_jobs)
    tasks = []
    for name, test in tests.items():
        if name in gts:
            tasks.append((name, gts[name], test))
        else:
            logging.warning("No ground truth for %s, skipping.", name)

    names = [name for name, _, _ in tasks]
    if n_jobs == 1 or len(tasks) < 2:
        accs = [_compare_dataframe_task(task, dist, distfields, distth) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            accs = list(executor.map(lambda task: _compare_dataframe_task(task, dist, distfields, distth), tasks))

    return accs, names


def _compare_single_sequence(gt, test, dist, distfields, distth, include_hota):
    if not include_hota:
        return utils.compare_to_groundtruth(gt, test, dist, distfields=distfields, distth=distth), None
    prepared = _prepare_iou_sequence_data(gt, test, distfields)
    return _compare_prepared_iou(prepared, distth), prepared


def _compare_multiple_sequences(gts, tests, dist, distfields, distth, include_hota, n_jobs):
    if include_hota:
        return _compare_iou_dataframes(gts, tests, distfields=distfields, distth=distth, n_jobs=n_jobs)
    accumulators, names = compare_dataframes(
        gts,
        tests,
        dist=dist,
        distfields=distfields,
        distth=distth,
        n_jobs=n_jobs,
    )
    return accumulators, names, None


def _compare_iou_dataframes(gts, tests, distfields=None, distth=0.5, n_jobs=1):
    """Prepare shared IoU data and build CLEAR accumulators."""
    _validate_n_jobs(n_jobs)
    tasks = []
    for name, test in tests.items():
        if name in gts:
            tasks.append((name, gts[name], test))
        else:
            logging.warning("No ground truth for %s, skipping.", name)

    def prepare_and_compare(task):
        name, ground_truth, tracker = task
        logging.info("Comparing %s...", name)
        prepared = _prepare_iou_sequence_data(ground_truth, tracker, distfields)
        return prepared, _compare_prepared_iou(prepared, distth)

    if n_jobs == 1 or len(tasks) < 2:
        results = [prepare_and_compare(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(prepare_and_compare, tasks))

    names = [name for name, _, _ in tasks]
    prepared_sequences = OrderedDict((name, result[0]) for name, result in zip(names, results))
    accumulators = [result[1] for result in results]
    return accumulators, names, prepared_sequences


def _append_hota_summary(
    summary,
    gts,
    tests,
    names,
    metric_host,
    dist,
    distfields,
    hota_alphas,
    generate_overall,
    n_jobs,
    prepared_sequences=None,
):
    hota_summary = _compute_hota_summary(
        gts,
        tests,
        names,
        metric_host,
        dist,
        distfields,
        hota_alphas,
        generate_overall,
        n_jobs,
        summary.index,
        prepared_sequences=prepared_sequences,
    )
    return pd.concat([summary, hota_summary], axis=1)


def _compute_hota_summary(
    gts,
    tests,
    names,
    metric_host,
    dist,
    distfields,
    hota_alphas,
    generate_overall,
    n_jobs,
    index,
    prepared_sequences=None,
):
    del metric_host, dist, n_jobs  # HOTA is validated as IoU-only by the caller.
    hota_alphas = np.asarray(hota_alphas, dtype=float)
    if prepared_sequences is None:
        sequence_summaries = OrderedDict(
            (name, _compute_hota_sequence_summary(gts[name], tests[name], distfields, hota_alphas)) for name in names
        )
    else:
        sequence_summaries = OrderedDict(
            (name, _compute_prepared_hota_sequence_summary(prepared_sequences[name], hota_alphas)) for name in names
        )
    if generate_overall:
        sequence_summaries["OVERALL"] = _combine_hota_sequence_summaries(sequence_summaries.values())

    rows = []
    for row_name in index:
        row = OrderedDict()
        for alpha_metric, summary_metric in HOTA_SUMMARY_METRICS.items():
            row[summary_metric] = np.mean(sequence_summaries[row_name][alpha_metric])
        rows.append(row)
    return pd.DataFrame(rows, index=index)


def _compute_hota_sequence_summary(gt, test, distfields, hota_alphas):
    prepared = _prepare_iou_sequence_data(gt, test, distfields)
    return _compute_prepared_hota_sequence_summary(prepared, hota_alphas)


def _compute_prepared_hota_sequence_summary(prepared, hota_alphas):
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
        if similarities.size == 0:
            continue
        weighted_similarities = similarities * alignment_scores[np.ix_(gt_indices, tracker_indices)]
        row_indices, col_indices = lap.linear_sum_assignment(1 - weighted_similarities)
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


def _prepare_iou_sequence_data(gt, test, distfields=None):
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

    for frame_id in frame_ids:
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
        # Preserve the legacy advanced-index behavior for invalid inputs that
        # repeat an identity within one frame.
        id_counts[frame_id_codes] += 1
    return groups, id_counts


def _compare_prepared_iou(prepared, distth):
    accumulator = MOTAccumulator()
    for frame_id, gt_indices, tracker_indices, similarities in prepared.frame_data:
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


def _compare_dataframe_task(task, dist, distfields, distth):
    name, gt, test = task
    logging.info("Comparing %s...", name)
    return utils.compare_to_groundtruth(gt, test, dist, distfields=distfields, distth=distth)


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
