# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Tests issue 19.

https://github.com/cheind/py-motmetrics/issues/19
"""

import numpy as np

import motmetrics._metrics as metrics
from motmetrics._accumulator import _Accumulator


def test_issue19():
    """Tests issue 19."""
    acc = _Accumulator(np.ones(4, dtype=int), np.ones(6, dtype=int))

    g0 = [0, 1]
    p0 = [0, 1]
    d0 = [[0.2, np.nan], [np.nan, 0.2]]

    g1 = [2, 3]
    p1 = [2, 3, 4, 5]
    d1 = [[0.28571429, 0.5, 0.0, np.nan], [np.nan, 0.44444444, np.nan, 0.0]]

    d0 = np.asarray(d0)
    d1 = np.asarray(d1)
    acc.update(np.asarray(g0), np.asarray(p0), d0, np.isfinite(d0))
    acc.update(np.asarray(g1), np.asarray(p1), d1, np.isfinite(d1))

    metrics._compute_metrics(acc)
