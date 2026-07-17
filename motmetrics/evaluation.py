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
        acc = utils.compare_to_groundtruth(gt, test, dist, distfields=distfields, distth=distth)

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
            )
    else:
        gt = load_motchallenge_groundtruths(gt_path, fmt=fmt, min_confidence=gt_min_confidence)
        test = load_motchallenge_tests(test_path, fmt=fmt)
        accs, names = compare_dataframes(gt, test, dist=dist, distfields=distfields, distth=distth, n_jobs=n_jobs)
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


def load_motchallenge_tests(root, fmt=io.Format.AUTO):
    """Load MOTChallenge tracker result dataframes from a file or folder."""
    root = Path(root)
    if root.is_file():
        return OrderedDict([(_default_sequence_name(root), io.loadtxt(root, fmt=fmt))])

    files = _find_test_files(root)
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


def _append_hota_summary(summary, gts, tests, names, metric_host, dist, distfields, hota_alphas, generate_overall, n_jobs):
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
    )
    return pd.concat([summary, hota_summary], axis=1)


def _compute_hota_summary(gts, tests, names, metric_host, dist, distfields, hota_alphas, generate_overall, n_jobs, index):
    hota_accumulators = _compare_hota_dataframes(gts, tests, names, dist, distfields, hota_alphas, n_jobs)
    per_alpha_summaries = []

    for alpha_index in range(len(hota_alphas)):
        per_alpha_summaries.append(
            metric_host.compute_many(
                [hota_accumulators[name][alpha_index] for name in names],
                metrics=HOTA_ALPHA_METRICS,
                names=names,
                generate_overall=generate_overall,
                n_jobs=n_jobs,
            )
        )

    rows = []
    for row_name in index:
        row = OrderedDict()
        for alpha_metric, summary_metric in HOTA_SUMMARY_METRICS.items():
            row[summary_metric] = np.mean([summary.loc[row_name, alpha_metric] for summary in per_alpha_summaries])
        rows.append(row)
    return pd.DataFrame(rows, index=index)


def _compare_hota_dataframes(gts, tests, names, dist, distfields, hota_alphas, n_jobs):
    tasks = [(name, gts[name], tests[name]) for name in names]
    if n_jobs == 1 or len(tasks) < 2:
        results = [_compare_hota_dataframe_task(task, dist, distfields, hota_alphas) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(lambda task: _compare_hota_dataframe_task(task, dist, distfields, hota_alphas), tasks))
    return OrderedDict(results)


def _compare_hota_dataframe_task(task, dist, distfields, hota_alphas):
    name, gt, test = task
    logging.info("Comparing %s for HOTA...", name)
    accs = utils.compare_to_groundtruth_reweighting(gt, test, dist, distfields=distfields, distth=hota_alphas)
    return name, accs


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
