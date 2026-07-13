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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from importlib.metadata import version as _distribution_version

__all__ = [
    "distances",
    "io",
    "lap",
    "metrics",
    "utils",
    "MOTAccumulator",
]

from motmetrics import distances
from motmetrics import io
from motmetrics import lap
from motmetrics import metrics
from motmetrics import utils
from motmetrics.mot import MOTAccumulator

__version__ = _distribution_version("motmetrics")
