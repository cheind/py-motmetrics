# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

from __future__ import absolute_import, division, print_function

from importlib.metadata import version as _distribution_version

__all__ = [
    "distances",
    "evaluation",
    "io",
    "lap",
    "metrics",
    "utils",
    "evaluate_motchallenge",
    "list_metrics",
    "list_metrics_markdown",
    "MOTChallengeSummary",
    "MOTAccumulator",
]

from motmetrics import distances, evaluation, io, lap, metrics, utils
from motmetrics.evaluation import MOTChallengeSummary, evaluate_motchallenge
from motmetrics.mot import MOTAccumulator

__version__ = _distribution_version("motmetrics")


def list_metrics(include_deps=False):
    """Return all registered metrics as a pandas DataFrame."""
    return metrics.create().list_metrics(include_deps=include_deps)


def list_metrics_markdown(include_deps=False):
    """Return all registered metrics as a markdown table."""
    return metrics.create().list_metrics_markdown(include_deps=include_deps)
