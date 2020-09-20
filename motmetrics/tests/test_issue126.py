# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Tests for issue 126.

https://github.com/cheind/py-motmetrics/issues/126
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import glob
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

import motmetrics as mm

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
SOLVERS = ['scipy', 'lap', 'lapsolver', 'ortools']

# https://github.com/dendorferpatrick/MOTChallengeEvalKit/blob/56aa36785a145184037412d45ed04b0c74c0ff89/matlab_devkit/utils/preprocessResult.m#L72
MOT_IGNORE_CATEGORIES = [
        2,  # person_on_vhcl
        7,  # static_person
        8,  # distractor
        12,  # reflection
        # 6,  # non_mot_vhcl (MOT20 only?)
]

@pytest.mark.parametrize('solver', SOLVERS)
def test_issue126(solver):
    """Checks that results match those of official devkit."""
    gt_dir = os.path.join(DATA_DIR, 'issue126', 'MOT17', 'train')
    pred_dir = os.path.join(DATA_DIR, 'issue126', 'Lif_T')

    gt_files = glob.glob(os.path.join(gt_dir, '*', 'gt', 'gt.txt'))
    pred_files = glob.glob(os.path.join(pred_dir, '*.txt'))
    gt_seqs = [pathlib.Path(f).parts[-3] for f in gt_files]
    pred_seqs = [os.path.splitext(pathlib.Path(f).parts[-1])[0] for f in pred_files]
    gt_files = dict(zip(gt_seqs, gt_files))
    pred_files = dict(zip(pred_seqs, pred_files))

    seqs = sorted(gt_seqs)
    gt_files = [gt_files[seq] for seq in seqs]
    pred_files = [pred_files[seq] for seq in seqs]

    gt = [mm.io.loadtxt(f, fmt='mot16') for f in gt_files]
    pred = [mm.io.loadtxt(f, fmt='mot16') for f in pred_files]
    pred_subset = [None for _ in pred_files]

    # Check whether there are some matches to other classes.
    num_gt = collections.Counter()
    num_pred = collections.Counter()
    FIELDS = ['X', 'Y', 'Width', 'Height']
    for i in range(len(seqs)):
        num_frames = gt[i].index.get_level_values('FrameId').max()
        frame_to_gt = dict(iter(gt[i].groupby('FrameId')))
        frame_to_pred = dict(iter(pred[i].groupby('FrameId')))
        frame_to_pred_subset = {}
        for t in range(1, num_frames + 1):
            if t in frame_to_gt:
                num_gt.update(frame_to_gt[t]['ClassId'])
            if t in frame_to_pred:
                # Assume all predictions add to class 1.
                num_pred[1] += len(frame_to_pred[t])

            if not (t in frame_to_gt and t in frame_to_pred):
                continue
            dists = mm.distances.iou_matrix(frame_to_gt[t][FIELDS].values, frame_to_pred[t][FIELDS].values, 0.5)
            is_match = ~np.isnan(dists)
            # is_pedestrian = np.asarray(frame_to_gt[t]['ClassId'] == 1)[:, np.newaxis]
            # weights = np.where(is_match, 1e3 * is_pedestrian + is_match + 1e-3 * (1 - dists), 0)
            weights = np.where(is_match, is_match + 1e-3 * (1 - dists), 0)
            gt_ind, pred_ind = mm.lap.linear_sum_assignment(-weights)
            non_zero_subset = is_match[gt_ind, pred_ind]
            gt_ind = gt_ind[non_zero_subset]
            pred_ind = pred_ind[non_zero_subset]
            gt_cat = frame_to_gt[t].iloc[gt_ind]['ClassId']

            # Find predictions to exclude.
            exclude_ind = pred_ind[gt_cat.isin(MOT_IGNORE_CATEGORIES)]
            exclude_mask = _subset_to_mask(len(frame_to_pred[t]), exclude_ind)
            frame_to_pred_subset[t] = frame_to_pred[t][~exclude_mask]

        # Note: Assumes that the predictions are not empty.
        pred_subset[i] = pd.concat([frame_to_pred_subset[t] for t in sorted(frame_to_pred_subset)])

    pred = pred_subset

    # Take only pedestrian class.
    gt = [df[df['ClassId'] == 1] for df in gt]

    expected = pd.Series(collections.OrderedDict([
        ('num_false_positives', 2655),
        ('num_misses', 107803),
        ('mostly_tracked', 679),
        ('partially_tracked', 595),
        ('mostly_lost', 364),
        ('num_fragmentations', 1153),
        ('num_switches', 791),
    ]))

    with mm.lap.set_default_solver(solver):
        accs = [mm.utils.compare_to_groundtruth(gt[i], pred[i], 'iou', distth=0.5) for i in range(len(seqs))]
        metrics = list(expected.index)
        mh = mm.metrics.create()
        summary = mh.compute_many(accs, names=seqs, metrics=metrics, generate_overall=True)

        overall = summary.loc['OVERALL']
        pd.testing.assert_series_equal(overall, expected, check_names=False)


def _subset_to_mask(n, subset):
    mask = np.zeros(n, dtype=np.bool)
    mask[subset] = True
    return mask
