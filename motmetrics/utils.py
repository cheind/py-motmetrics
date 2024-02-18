# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Functions for populating event accumulators."""

from __future__ import absolute_import, division, print_function

import numpy as np

from motmetrics.distances import iou_matrix, norm2squared_matrix
from motmetrics.mot import MOTAccumulator
from motmetrics.preprocess import preprocessResult


def compute_global_aligment_score(
    allframeids, fid_to_fgt, fid_to_fdt, num_gt_id, num_det_id, dist_func
):
    """Taken from https://github.com/JonathonLuiten/TrackEval/blob/12c8791b303e0a0b50f753af204249e622d0281a/trackeval/metrics/hota.py"""
    potential_matches_count = np.zeros((num_gt_id, num_det_id))
    gt_id_count = np.zeros((num_gt_id, 1))
    tracker_id_count = np.zeros((1, num_det_id))

    for fid in allframeids:
        oids = np.empty(0)
        hids = np.empty(0)
        if fid in fid_to_fgt:
            fgt = fid_to_fgt[fid]
            oids = fgt.index.get_level_values("Id")
        if fid in fid_to_fdt:
            fdt = fid_to_fdt[fid]
            hids = fdt.index.get_level_values("Id")
        if len(oids) > 0 and len(hids) > 0:
            gt_ids = np.array(oids.values) - 1
            dt_ids = np.array(hids.values) - 1
            similarity = dist_func(fgt.values, fdt.values, return_dist=False)

            sim_iou_denom = (
                similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            )
            sim_iou = np.zeros_like(similarity)
            sim_iou_mask = sim_iou_denom > 0 + np.finfo("float").eps
            sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
            potential_matches_count[gt_ids[:, np.newaxis], dt_ids[np.newaxis, :]] += sim_iou

            # Calculate the total number of dets for each gt_id and tracker_id.
            gt_id_count[gt_ids] += 1
            tracker_id_count[0, dt_ids] += 1
    global_alignment_score = potential_matches_count / (
        np.maximum(1, gt_id_count + tracker_id_count - potential_matches_count)
    )
    return global_alignment_score


def compare_to_groundtruth_reweighting(gt, dt, dist="iou", distfields=None, distth=(0.5)):
    """Compare groundtruth and detector results with global alignment score.

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
    distth: Union(float, array_like), optional
        Maximum tolerable distance. Pairs exceeding this threshold are marked 'do-not-pair'.
        If a list of thresholds is given, multiple accumulators are returned.
    """
    # pylint: disable=too-many-locals
    if distfields is None:
        distfields = ["X", "Y", "Width", "Height"]

    def compute_iou(a, b, return_dist):
        return iou_matrix(a, b, max_iou=distth, return_dist=return_dist)

    def compute_euc(a, b, *args, **kwargs):
        return np.sqrt(norm2squared_matrix(a, b, max_d2=distth**2))

    def compute_seuc(a, b, *args, **kwargs):
        return norm2squared_matrix(a, b, max_d2=distth)

    if dist.upper() == "IOU":
        compute_dist = compute_iou
    elif dist.upper() == "EUC":
        compute_dist = compute_euc
        import warnings

        warnings.warn(
            f"'euc' flag changed its behavior. The euclidean distance is now used instead of the squared euclidean distance. Make sure the used threshold (distth={distth}) is not squared. Use 'euclidean' flag to avoid this warning."
        )
    elif dist.upper() == "EUCLIDEAN":
        compute_dist = compute_euc
    elif dist.upper() == "SEUC":
        compute_dist = compute_seuc
    else:
        raise f'Unknown distance metric {dist}. Use "IOU", "EUCLIDEAN",  or "SEUC"'

    return_single = False
    if isinstance(distth, float):
        distth = [distth]
        return_single = True

    acc_list = [MOTAccumulator() for _ in range(len(distth))]

    num_gt_id = gt.index.get_level_values("Id").max()
    num_det_id = dt.index.get_level_values("Id").max()

    # We need to account for all frames reported either by ground truth or
    # detector. In case a frame is missing in GT this will lead to FPs, in
    # case a frame is missing in detector results this will lead to FNs.
    allframeids = gt.index.union(dt.index).levels[0]

    gt = gt[distfields]
    dt = dt[distfields]
    fid_to_fgt = dict(iter(gt.groupby("FrameId")))
    fid_to_fdt = dict(iter(dt.groupby("FrameId")))

    global_alignment_score = compute_global_aligment_score(
        allframeids, fid_to_fgt, fid_to_fdt, num_gt_id, num_det_id, compute_dist
    )

    for fid in allframeids:
        oids = np.empty(0)
        hids = np.empty(0)
        weighted_dists = np.empty((0, 0))
        if fid in fid_to_fgt:
            fgt = fid_to_fgt[fid]
            oids = fgt.index.get_level_values("Id")
        if fid in fid_to_fdt:
            fdt = fid_to_fdt[fid]
            hids = fdt.index.get_level_values("Id")
        if len(oids) > 0 and len(hids) > 0:
            gt_ids = np.array(oids.values) - 1
            dt_ids = np.array(hids.values) - 1
            dists = compute_dist(fgt.values, fdt.values, return_dist=False)
            weighted_dists = (
                dists * global_alignment_score[gt_ids[:, np.newaxis], dt_ids[np.newaxis, :]]
            )
        for acc, th in zip(acc_list, distth):
            acc.update(oids, hids, 1 - weighted_dists, frameid=fid, similartiy_matrix=dists, th=th)
    return acc_list[0] if return_single else acc_list


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
