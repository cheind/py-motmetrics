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
from motmetrics.lap import linear_sum_assignment
from motmetrics.mot import MOTAccumulator
from motmetrics.preprocess import preprocessResult


def compute_global_aligment_score(
    allframeids,
    fid_to_fgt,
    fid_to_fdt,
    num_gt_id,
    num_det_id,
    dist_func,
    return_details=False,
    gt_id_map=None,
    tracker_id_map=None,
):
    """Compute HOTA global alignment and optionally retain reusable frame data.

    Adapted from TrackEval's HOTA implementation:
    https://github.com/JonathonLuiten/TrackEval/blob/12c8791b303e0a0b50f753af204249e622d0281a/trackeval/metrics/hota.py
    """
    potential_matches_count = np.zeros((num_gt_id, num_det_id))
    gt_id_count = np.zeros(num_gt_id, dtype=int)
    tracker_id_count = np.zeros(num_det_id, dtype=int)
    frame_data = []

    for fid in allframeids:
        oids = np.empty(0, dtype=int)
        hids = np.empty(0, dtype=int)
        fgt = fid_to_fgt.get(fid)
        fdt = fid_to_fdt.get(fid)
        if fgt is not None:
            oids = fgt.index.get_level_values("Id").to_numpy()
        if fdt is not None:
            hids = fdt.index.get_level_values("Id").to_numpy()
        if gt_id_map is None:
            gt_ids = np.asarray(oids, dtype=int) - 1
        else:
            gt_ids = np.fromiter((gt_id_map[oid] for oid in oids), dtype=int)
        if tracker_id_map is None:
            dt_ids = np.asarray(hids, dtype=int) - 1
        else:
            dt_ids = np.fromiter((tracker_id_map[hid] for hid in hids), dtype=int)
        np.add.at(gt_id_count, gt_ids, 1)
        np.add.at(tracker_id_count, dt_ids, 1)

        similarity = np.empty((len(oids), len(hids)), dtype=float)
        if len(oids) > 0 and len(hids) > 0:
            similarity = dist_func(fgt.values, fdt.values, return_dist=False)

            sim_iou_denom = (
                similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            )
            sim_iou = np.zeros_like(similarity)
            sim_iou_mask = sim_iou_denom > np.finfo("float").eps
            sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
            potential_matches_count[gt_ids[:, np.newaxis], dt_ids[np.newaxis, :]] += sim_iou

        if return_details:
            frame_data.append((fid, oids, hids, gt_ids, dt_ids, similarity))

    global_alignment_score = potential_matches_count / np.maximum(
        1,
        gt_id_count[:, np.newaxis] + tracker_id_count[np.newaxis, :] - potential_matches_count,
    )
    if return_details:
        return global_alignment_score, frame_data, gt_id_count, tracker_id_count
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

    return_single = np.isscalar(distth)
    thresholds = np.atleast_1d(np.asarray(distth, dtype=float))

    gt_ids = gt.index.get_level_values("Id").unique()
    tracker_ids = dt.index.get_level_values("Id").unique()
    gt_id_map = {oid: index for index, oid in enumerate(gt_ids)}
    tracker_id_map = {hid: index for index, hid in enumerate(tracker_ids)}
    num_gt_id = len(gt_id_map)
    num_det_id = len(tracker_id_map)

    # We need to account for all frames reported either by ground truth or
    # detector. In case a frame is missing in GT this will lead to FPs, in
    # case a frame is missing in detector results this will lead to FNs.
    allframeids = gt.index.union(dt.index).levels[0]

    gt = gt[distfields]
    dt = dt[distfields]
    fid_to_fgt = dict(iter(gt.groupby("FrameId")))
    fid_to_fdt = dict(iter(dt.groupby("FrameId")))

    global_alignment_score, frame_data, gt_id_count, tracker_id_count = compute_global_aligment_score(
        allframeids,
        fid_to_fgt,
        fid_to_fdt,
        num_gt_id,
        num_det_id,
        compute_dist,
        return_details=True,
        gt_id_map=gt_id_map,
        tracker_id_map=tracker_id_map,
    )
    match_counts = np.zeros(
        (len(thresholds), num_gt_id, num_det_id),
        dtype=np.int64,
    )
    num_detections = np.zeros(len(thresholds), dtype=np.int64)
    deferred_frames = []
    threshold_epsilon = np.finfo("float").eps

    for fid, oids, hids, gt_ids, dt_ids, similarity in frame_data:
        weighted_similarity = np.empty_like(similarity)
        if len(oids) > 0 and len(hids) > 0:
            weighted_similarity = (
                similarity
                * global_alignment_score[gt_ids[:, np.newaxis], dt_ids[np.newaxis, :]]
            )
        matching_costs = 1 - weighted_similarity
        assignment = linear_sum_assignment(matching_costs)
        deferred_frames.append((oids, hids, fid, similarity, assignment))

        rows, columns = assignment
        if not rows.size:
            continue
        assigned_similarity = similarity[rows, columns]
        valid_threshold, valid_match = np.where(
            assigned_similarity[np.newaxis, :]
            >= thresholds[:, np.newaxis] - threshold_epsilon
        )
        matched_gt_ids = gt_ids[rows[valid_match]]
        matched_dt_ids = dt_ids[columns[valid_match]]
        np.add.at(
            match_counts,
            (valid_threshold, matched_gt_ids, matched_dt_ids),
            1,
        )
        num_detections += np.bincount(valid_threshold, minlength=len(thresholds))

    num_objects = int(gt_id_count.sum())
    num_predictions = int(tracker_id_count.sum())
    acc_list = []
    for threshold_index, threshold in enumerate(thresholds):
        detections = int(num_detections[threshold_index])
        false_positives = num_predictions - detections
        counts = match_counts[threshold_index]
        association = counts / np.maximum(
            1,
            gt_id_count[:, np.newaxis] + tracker_id_count[np.newaxis, :] - counts,
        )
        assa = (association * counts).sum() / max(1, detections)
        deta = detections / max(1, num_objects + false_positives)
        stats = {
            "num_detections": detections,
            "num_objects": num_objects,
            "num_false_positives": false_positives,
            "deta_alpha": deta,
            "assa_alpha": assa,
            "hota_alpha": (deta * assa) ** 0.5,
        }
        accumulator = MOTAccumulator()
        accumulator._defer_hota_event_updates(deferred_frames, threshold, stats)
        acc_list.append(accumulator)
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
