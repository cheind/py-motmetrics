# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Fast MOTChallenge evaluation."""

from importlib.metadata import version as _distribution_version

from motmetrics._evaluation import evaluate_motchallenge

__all__ = ["evaluate_motchallenge"]

__version__ = _distribution_version("motmetrics")
