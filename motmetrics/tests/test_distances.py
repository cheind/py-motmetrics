# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre, Mikel Broström and others.
# See LICENSE file for terms.

"""Tests distance computation."""

import numpy as np

import motmetrics._distances as distance_functions


def test_iou_matrix():
    """Tests iou_matrix."""
    a = np.array([
        [0, 0, 1, 2],
    ])

    b = np.array([
        [0, 0, 1, 2],
        [0, 0, 1, 1],
        [1, 1, 1, 1],
        [0.5, 0, 1, 1],
        [0, 1, 1, 1],
    ])
    np.testing.assert_allclose(
        distance_functions.iou_matrix(a, b),
        [[0, 0.5, 1, 0.8, 0.5]],
        atol=1e-4
    )

    np.testing.assert_allclose(
        distance_functions.iou_matrix(a, b, max_iou=0.5),
        [[0, 0.5, np.nan, np.nan, 0.5]],
        atol=1e-4
    )
