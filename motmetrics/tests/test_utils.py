# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Tests accumulation of events using utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import pandas as pd

import motmetrics as mm


def test_annotations_xor_predictions_present():
    """Tests frames that contain only annotations or predictions."""
    _ = None
    anno_tracks = {
        1: [0, 2, 4, 6, _, _, _],
        2: [_, _, 0, 2, 4, _, _],
    }
    pred_tracks = {
        1: [_, _, 3, 5, 7, 7, 7],
    }
    anno = _tracks_to_dataframe(anno_tracks)
    pred = _tracks_to_dataframe(pred_tracks)
    acc = mm.utils.compare_to_groundtruth(anno, pred, 'euc', distfields=['Position'], distth=2)
    mh = mm.metrics.create()
    metrics = mh.compute(acc, return_dataframe=False, metrics=[
        'num_objects', 'num_predictions', 'num_unique_objects',
    ])
    np.testing.assert_equal(metrics['num_objects'], 7)
    np.testing.assert_equal(metrics['num_predictions'], 5)
    np.testing.assert_equal(metrics['num_unique_objects'], 2)


def _tracks_to_dataframe(tracks):
    rows = []
    for track_id, track in tracks.items():
        for frame_id, position in zip(itertools.count(1), track):
            if position is None:
                continue
            rows.append({
                'FrameId': frame_id,
                'Id': track_id,
                'Position': position,
            })
    return pd.DataFrame(rows).set_index(['FrameId', 'Id'])
