"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

if __name__ == '__main__':
    import motmetrics

    mh = motmetrics.metrics.create()
    print(mh.list_metrics_markdown())
