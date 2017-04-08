"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

import numpy as np

def norm2squared_matrix(objs, hyps, max_d2=float('inf')):
    """Computes the distance matrix between object and hypothesis points using squared Euclidean distance."""

    objs = np.atleast_2d(objs).astype(float)
    hyps = np.atleast_2d(hyps).astype(float)

    if objs.size == 0 or hyps.size == 0:
        return np.empty((0,0))
    
    C = np.empty((objs.shape[0], hyps.shape[0]))
        
    for o in range(objs.shape[0]):
        for h in range(hyps.shape[0]):
            e = objs[o] - hyps[h]
            C[o, h] = e.dot(e)

    C[C > max_d2] = np.nan
    return C


def iou_matrix(objs, hyps, max_iou=1.):
    """Computes the distance matrix between object and hypothesis rectangles using intersection over union distance."""

    objs = np.atleast_2d(objs).astype(float)
    hyps = np.atleast_2d(hyps).astype(float)

    if objs.size == 0 or hyps.size == 0:
        return np.empty((0,0))

    assert objs.shape[1] == 4
    assert hyps.shape[1] == 4

    br_objs = objs[:, :2] + objs[:, 2:]
    br_hyps = hyps[:, :2] + hyps[:, 2:]

    C = np.empty((objs.shape[0], hyps.shape[0]))

    for o in range(objs.shape[0]):
        for h in range(hyps.shape[0]):
            isect_xy = np.maximum(objs[o, :2], hyps[h, :2])
            isect_wh = np.maximum(np.minimum(br_objs[o], br_hyps[h]) - isect_xy, 0)
            isect_a = isect_wh[0]*isect_wh[1]
            union_a = objs[o, 2]*objs[o, 3] + hyps[h, 2]*hyps[h, 3] - isect_a
            C[o, h] = 1. - isect_a / union_a

    C[C > max_iou] = np.nan
    return C
    





