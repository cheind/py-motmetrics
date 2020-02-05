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

"""List metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

if __name__ == '__main__':
    import motmetrics

    mh = motmetrics.metrics.create()
    print(mh.list_metrics_markdown())
