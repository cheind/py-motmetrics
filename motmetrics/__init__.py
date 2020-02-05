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

"""py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = [
    'distances',
    'io',
    'lap',
    'metrics',
    'utils',
    'MOTAccumulator',
]

from motmetrics import distances
from motmetrics import io
from motmetrics import lap
from motmetrics import metrics
from motmetrics import utils
from motmetrics.mot import MOTAccumulator

# Needs to be last line
__version__ = '1.1.3'
