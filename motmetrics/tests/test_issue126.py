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

import pandas as pd
import pytest

import motmetrics as mm

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
SOLVERS = ['scipy', 'lap', 'lapsolver', 'ortools']


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
