# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Functions for comparing predictions and ground-truth."""

import numpy as np


def iou_matrix(objs, hyps, max_iou=1., return_dist=True):
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
    return_dist : bool
        If true, return distance matrix. If false, return similarity (IoU) matrix.

    Returns
    -------
    C : NxK array
        Distance matrix containing pairwise distances or np.nan.
        if `return_dist` is False, then the matrix contains the pairwise IoU.
    """

    if np.size(objs) == 0 or np.size(hyps) == 0:
        return np.empty((0, 0))

    objs = np.asarray(objs, dtype=float)
    hyps = np.asarray(hyps, dtype=float)
    assert objs.shape[1] == 4
    assert hyps.shape[1] == 4
    # Compute the dominant evaluation inner loop directly, without generic
    # broadcast coordinate/size temporaries.
    intersection_width = np.maximum(
        np.minimum(objs[:, None, 0] + objs[:, None, 2], hyps[None, :, 0] + hyps[None, :, 2])
        - np.maximum(objs[:, None, 0], hyps[None, :, 0]),
        0.0,
    )
    intersection_height = np.maximum(
        np.minimum(objs[:, None, 1] + objs[:, None, 3], hyps[None, :, 1] + hyps[None, :, 3])
        - np.maximum(objs[:, None, 1], hyps[None, :, 1]),
        0.0,
    )
    intersection = intersection_width * intersection_height
    object_areas = np.maximum(objs[:, 2], 0.0) * np.maximum(objs[:, 3], 0.0)
    hypothesis_areas = np.maximum(hyps[:, 2], 0.0) * np.maximum(hyps[:, 3], 0.0)
    union = object_areas[:, None] + hypothesis_areas[None, :] - intersection
    iou = np.zeros_like(intersection)
    np.divide(intersection, union, out=iou, where=intersection != 0.0)
    if return_dist:
        dist = 1 - iou
        return np.where(dist > max_iou, np.nan, dist)
    return iou
