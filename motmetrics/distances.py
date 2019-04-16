"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
Toka, 2018
https://github.com/cheind/py-motmetrics
Fast implement by TOKA
"""

import numpy as np

def norm2squared_matrix(objs, hyps, max_d2=float('inf')):
    """Computes the squared Euclidean distance matrix between object and hypothesis points.
    
    Params
    ------
    objs : NxM array
        Object points of dim M in rows
    hyps : KxM array
        Hypothesis points of dim M in rows

    Kwargs
    ------
    max_d2 : float
        Maximum tolerable squared Euclidean distance. Object / hypothesis points
        with larger distance are set to np.nan signalling do-not-pair. Defaults
        to +inf

    Returns
    -------
    C : NxK array
        Distance matrix containing pairwise distances or np.nan.
    """

    objs = np.atleast_2d(objs).astype(float)
    hyps = np.atleast_2d(hyps).astype(float)

    if objs.size == 0 or hyps.size == 0:
        return np.empty((0,0))

    assert hyps.shape[1] == objs.shape[1], "Dimension mismatch"
    
    C = np.empty((objs.shape[0], hyps.shape[0]))
        
    for o in range(objs.shape[0]):
        for h in range(hyps.shape[0]):
            e = objs[o] - hyps[h]
            C[o, h] = e.dot(e)

    C[C > max_d2] = np.nan
    return C

def boxiou(a, b):
    rx1 = max(a[0], b[0])
    rx2 = min(a[0]+a[2], b[0]+b[2])
    ry1 = max(a[1], b[1])
    ry2 = min(a[1]+a[3], b[1]+b[3])
    if ry2>ry1 and rx2>rx1:
        i = (ry2-ry1)*(rx2-rx1)
        u = a[2]*a[3]+b[2]*b[3]-i
        return float(i)/u
    else: return 0.0

def iou_matrix(objs, hyps, max_iou=1.):
    """Computes 'intersection over union (IoU)' distance matrix between object and hypothesis rectangles.

    The IoU is computed as 
        
        IoU(a,b) = 1. - isect(a, b) / union(a, b)

    where isect(a,b) is the area of intersection of two rectangles and union(a, b) the area of union. The
    IoU is bounded between zero and one. 0 when the rectangles overlap perfectly and 1 when the overlap is
    zero.
    
    Params
    ------
    objs : Nx4 array
        Object rectangles (x,y,w,h) in rows
    hyps : Kx4 array
        Hypothesis rectangles (x,y,w,h) in rows

    Kwargs
    ------
    max_iou : float
        Maximum tolerable overlap distance. Object / hypothesis points
        with larger distance are set to np.nan signalling do-not-pair. Defaults
        to 0.5

    Returns
    -------
    C : NxK array
        Distance matrix containing pairwise distances or np.nan.
    """

    #import time
    #st = time.time()
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
            #isect_xy = np.maximum(objs[o, :2], hyps[h, :2])
            #isect_wh = np.maximum(np.minimum(br_objs[o], br_hyps[h]) - isect_xy, 0)
            #isect_a = isect_wh[0]*isect_wh[1]
            #union_a = objs[o, 2]*objs[o, 3] + hyps[h, 2]*hyps[h, 3] - isect_a
            #if union_a != 0:
            #    C[o, h] = 1. - isect_a / union_a
            #else:
            #    C[o, h] = np.nan
            iou = boxiou(objs[o], hyps[h])
            if 1 - iou > max_iou:
                C[o, h] = np.nan
            else:
                C[o, h] = 1 - iou

    #C[C > max_iou] = np.nan
    #print('----'*2,'done',time.time()-st)
    return C
    





