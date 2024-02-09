# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Functions for populating event accumulators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from motmetrics.distances import iou_matrix, norm2squared_matrix
from motmetrics.mot import MOTAccumulator
from motmetrics.preprocess import preprocessResult


def compare_to_groundtruth(gt, dt, dist='iou', distfields=None, distth=0.5):
    """Compare groundtruth and detector results.

    This method assumes both results are given in terms of DataFrames with at least the following fields
     - `FrameId` First level index used for matching ground-truth and test frames.
     - `Id` Secondary level index marking available object / hypothesis ids

    Depending on the distance to be used relevant distfields need to be specified.

    Params
    ------
    gt : pd.DataFrame
        Dataframe for ground-truth
    test : pd.DataFrame
        Dataframe for detector results

    Kwargs
    ------
    dist : str, optional
        String identifying distance to be used. Defaults to intersection over union ('iou'). Euclidean
        distance ('euclidean') and squared euclidean distance ('seuc') are also supported.
    distfields: array, optional
        Fields relevant for extracting distance information. Defaults to ['X', 'Y', 'Width', 'Height']
    distth: float, optional
        Maximum tolerable distance. Pairs exceeding this threshold are marked 'do-not-pair'.
    """
    # pylint: disable=too-many-locals
    if distfields is None:
        distfields = ['X', 'Y', 'Width', 'Height']

    def compute_iou(a, b):
        return iou_matrix(a, b, max_iou=distth)

    def compute_euc(a, b):
        return np.sqrt(norm2squared_matrix(a, b, max_d2=distth**2))

    def compute_seuc(a, b):
        return norm2squared_matrix(a, b, max_d2=distth)

    if dist.upper() == 'IOU':
        compute_dist = compute_iou
    elif dist.upper() == 'EUC':
        compute_dist = compute_euc
        import warnings
        warnings.warn(f"'euc' flag changed its behavior. The euclidean distance is now used instead of the squared euclidean distance. Make sure the used threshold (distth={distth}) is not squared. Use 'euclidean' flag to avoid this warning.")
    elif dist.upper() == 'EUCLIDEAN':
        compute_dist = compute_euc
    elif dist.upper() == 'SEUC':
        compute_dist = compute_seuc
    else:
        raise f'Unknown distance metric {dist}. Use "IOU", "EUCLIDEAN",  or "SEUC"'

    acc = MOTAccumulator()

    # We need to account for all frames reported either by ground truth or
    # detector. In case a frame is missing in GT this will lead to FPs, in
    # case a frame is missing in detector results this will lead to FNs.
    allframeids = gt.index.union(dt.index).levels[0]

    gt = gt[distfields]
    dt = dt[distfields]
    fid_to_fgt = dict(iter(gt.groupby('FrameId')))
    fid_to_fdt = dict(iter(dt.groupby('FrameId')))

    for fid in allframeids:
        oids = np.empty(0)
        hids = np.empty(0)
        dists = np.empty((0, 0))
        if fid in fid_to_fgt:
            fgt = fid_to_fgt[fid]
            oids = fgt.index.get_level_values('Id')
        if fid in fid_to_fdt:
            fdt = fid_to_fdt[fid]
            hids = fdt.index.get_level_values('Id')
        if len(oids) > 0 and len(hids) > 0:
            dists = compute_dist(fgt.values, fdt.values)
        acc.update(oids, hids, dists, frameid=fid)

    return acc


def CLEAR_MOT_M(gt, dt, inifile, dist='iou', distfields=None, distth=0.5, include_all=False, vflag=''):
    """Compare groundtruth and detector results.

    This method assumes both results are given in terms of DataFrames with at least the following fields
     - `FrameId` First level index used for matching ground-truth and test frames.
     - `Id` Secondary level index marking available object / hypothesis ids

    Depending on the distance to be used relevant distfields need to be specified.

    Params
    ------
    gt : pd.DataFrame
        Dataframe for ground-truth
    test : pd.DataFrame
        Dataframe for detector results

    Kwargs
    ------
    dist : str, optional
        String identifying distance to be used. Defaults to intersection over union.
    distfields: array, optional
        Fields relevant for extracting distance information. Defaults to ['X', 'Y', 'Width', 'Height']
    distth: float, optional
        Maximum tolerable distance. Pairs exceeding this threshold are marked 'do-not-pair'.
    """
    # pylint: disable=too-many-locals
    if distfields is None:
        distfields = ['X', 'Y', 'Width', 'Height']

    def compute_iou(a, b):
        return iou_matrix(a, b, max_iou=distth)

    def compute_euc(a, b):
        return norm2squared_matrix(a, b, max_d2=distth)

    compute_dist = compute_iou if dist.upper() == 'IOU' else compute_euc

    acc = MOTAccumulator()
    dt = preprocessResult(dt, gt, inifile)
    if include_all:
        gt = gt[gt['Confidence'] >= 0.99]
    else:
        gt = gt[(gt['Confidence'] >= 0.99) & (gt['ClassId'] == 1)]
    # We need to account for all frames reported either by ground truth or
    # detector. In case a frame is missing in GT this will lead to FPs, in
    # case a frame is missing in detector results this will lead to FNs.
    allframeids = gt.index.union(dt.index).levels[0]
    analysis = {'hyp': {}, 'obj': {}}
    for fid in allframeids:
        oids = np.empty(0)
        hids = np.empty(0)
        dists = np.empty((0, 0))

        if fid in gt.index:
            fgt = gt.loc[fid]
            oids = fgt.index.values
            for oid in oids:
                oid = int(oid)
                if oid not in analysis['obj']:
                    analysis['obj'][oid] = 0
                analysis['obj'][oid] += 1

        if fid in dt.index:
            fdt = dt.loc[fid]
            hids = fdt.index.values
            for hid in hids:
                hid = int(hid)
                if hid not in analysis['hyp']:
                    analysis['hyp'][hid] = 0
                analysis['hyp'][hid] += 1

        if oids.shape[0] > 0 and hids.shape[0] > 0:
            dists = compute_dist(fgt[distfields].values, fdt[distfields].values)

        acc.update(oids, hids, dists, frameid=fid, vf=vflag)

    return acc, analysis


def is_in_region(bbox, reg):
    # Check if the 4 points of the bbox are inside region
    points = []
    # Center
    cx = bbox[0] + (bbox[2] / 2)
    cy = bbox[1] + (bbox[3] / 2)
    points.append(Point(cx, cy))
    # # Top-left
    # x1 = bbox[0]
    # y1 = bbox[1]
    # points.append(Point(x1, y1))
    # # Top-right
    # x1 = bbox[0] + bbox[2]
    # y1 = bbox[1]
    # points.append(Point(x1, y1))
    # # Bot-right
    # x1 = bbox[0] + bbox[2]
    # y1 = bbox[1] + bbox[3]
    # points.append(Point(x1, y1))
    # # Bot-left
    # x1 = bbox[0]
    # y1 = bbox[1] + bbox[3]
    # points.append(Point(x1, y1))

    # Region
    p_xy0 = (reg[0], reg[1])
    p_xy1 = (reg[0] + reg[2], reg[1])
    p_xy2 = (reg[0] + reg[2], reg[1] + reg[3])
    p_xy3 = (reg[0], reg[1] + reg[3])
    region = [p_xy0, p_xy1, p_xy2, p_xy3]
    polygon = Polygon(region)

    flags_inside = [polygon.contains(p) for p in points]
    flag_inside = all(flags_inside)

    return flag_inside
