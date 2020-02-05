# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
#
# Copyright (c) 2017-2020 Christoph Heindl
# Copyright (c) 2018 Toka
# Copyright (c) 2019-2020 Jack Valmadre
#
# See LICENSE file.

"""Math utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import numpy as np


def quiet_divide(a, b):
    """Quiet divide function that does not warn about (0 / 0)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.true_divide(a, b)
