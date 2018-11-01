import numpy as np
import motmetrics as mm

def test_issue19():
    acc = mm.MOTAccumulator()

    g0 = [0, 1]
    p0 = [0, 1]
    d0 = [[0.2, np.nan], [np.nan, 0.2]]

    g1 = [2, 3]
    p1 = [2, 3, 4, 5]
    d1 = [[0.28571429, 0.5, 0.0, np.nan], [ np.nan, 0.44444444, np.nan, 0.0 ]]

    acc.update(g0, p0, d0, 0)
    acc.update(g1, p1, d1, 1)

    mh = mm.metrics.create()
    result = mh.compute(acc)