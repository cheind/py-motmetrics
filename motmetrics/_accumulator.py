# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Accumulate compact tracking metric state frame by frame."""

import numpy as np

from motmetrics._assignment import _linear_sum_assignment
from motmetrics._metrics_engine import _MetricsEngine


class _Accumulator(object):
    """Match frames while incrementally updating compact metric primitives.

    Call :meth:`update` once per frame with ground-truth IDs, prediction IDs,
    and their pairwise distance matrix.

    References
    ----------
    1. Bernardin, Keni, and Rainer Stiefelhagen. "Evaluating multiple object tracking performance: the CLEAR MOT metrics."
    EURASIP Journal on Image and Video Processing 2008.1 (2008): 1-10.
    2. Milan, Anton, et al. "Mot16: A benchmark for multi-object tracking." arXiv preprint arXiv:1603.00831 (2016).
    3. Li, Yuan, Chang Huang, and Ram Nevatia. "Learning to associate: Hybridboosted multi-target tracker for crowded scene."
    Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on. IEEE, 2009.
    """

    def __init__(self, auto_id=False, max_switch_time=float('inf')):
        """Create a _Accumulator.

        Params
        ------
        auto_id : bool, optional
            Whether or not frame indices are auto-incremented or provided upon
            updating. Defaults to false. Not specifying a frame-id when this value
            is true results in an error. Specifying a frame-id when this value is
            false also results in an error.

        max_switch_time : scalar, optional
            Allows specifying an upper bound on the timespan an unobserved but
            tracked object is allowed to generate track switch events. Useful if groundtruth
            objects leaving the field of view keep their ID when they reappear,
            but your tracker is not capable of recognizing this (resulting in
            track switch events). The default is that there is no upper bound
            on the timespan. In units of frame timestamps. When using auto_id
            in units of count.

        """

        # Parameters of the accumulator.
        self.auto_id = auto_id
        self.max_switch_time = max_switch_time

        self.reset()

    def reset(self):
        """Reset the accumulator to empty state."""

        self._object_assignments = {}
        self._hypothesis_assignments = {}
        self._object_last_seen = {}
        self._object_last_matched = {}
        self._hypothesis_last_seen = {}
        self._last_frame_id = None
        self._metrics_engine = _MetricsEngine()

    def _record_event(self, typestr, oid, distance):
        """Record one event in the compact metric engine."""
        self._metrics_engine.record_event(typestr, oid, distance)

    def _record_event_batch(self, typestr, count, oids=None):
        """Record a batch of one event type in the compact metric engine."""
        if count == 0:
            return
        self._metrics_engine.record_event_batch(typestr, count, oids=oids)

    def update(self, oids, hids, dists, frameid=None):  # noqa: C901
        """Updates the accumulator with frame specific objects/detections.

        This method generates events based on the following algorithm [1]:
        1. Try to carry forward already established tracks. If any paired object / hypothesis
        from previous timestamps are still visible in the current frame, create a 'MATCH'
        event between them.
        2. For the remaining constellations minimize the total object / hypothesis distance
        error (Kuhn-Munkres algorithm). If a correspondence made contradicts a previous
        match create a 'SWITCH' else a 'MATCH' event.
        3. Create 'MISS' events for all remaining unassigned objects.
        4. Create 'FP' events for all remaining unassigned hypotheses.

        Params
        ------
        oids : N array
            Array of object ids.
        hids : M array
            Array of hypothesis ids.
        dists: NxM array
            Distance matrix. np.nan values to signal do-not-pair constellations.
            See `distances` module for support methods.

        Kwargs
        ------
        frameId : id
            Unique frame id. Optional when _Accumulator.auto_id is specified during
            construction.
        Returns
        -------
        frame_id
            The frame identifier that was processed.

        References
        ----------
        1. Bernardin, Keni, and Rainer Stiefelhagen. "Evaluating multiple object tracking performance: the CLEAR MOT metrics."
        EURASIP Journal on Image and Video Processing 2008.1 (2008): 1-10.
        """
        # pylint: disable=too-many-locals, too-many-statements

        oids = np.asarray(oids)
        oids_masked = np.zeros_like(oids, dtype=np.bool_)
        hids = np.asarray(hids)
        hids_masked = np.zeros_like(hids, dtype=np.bool_)
        dists = np.atleast_2d(dists).astype(float).reshape(oids.shape[0], hids.shape[0]).copy()

        if frameid is None:
            assert self.auto_id, 'auto-id is not enabled'
            if self._last_frame_id is not None:
                frameid = self._last_frame_id + 1
            else:
                frameid = 0
        else:
            assert not self.auto_id, 'Cannot provide frame id when auto-id is enabled'

        # 0. Record the frame and all finite object/hypothesis pairs.
        valid_i, valid_j = np.where(np.isfinite(dists))
        self._metrics_engine.record_frame(oids, hids, valid_i, valid_j)

        if oids.size * hids.size > 0:
            # 1. Try to re-establish tracks from correspondences in last update.
            for i in range(oids.shape[0]):
                if not (
                    oids[i] in self._object_assignments
                    and self._object_last_matched[oids[i]] == self._last_frame_id
                ):
                    continue

                hprev = self._object_assignments[oids[i]]
                (candidate_indices,) = np.where(~hids_masked & (hids == hprev))
                if candidate_indices.shape[0] == 0:
                    continue
                j = candidate_indices[0]

                if np.isfinite(dists[i, j]):
                    o = oids[i]
                    h = hids[j]
                    oids_masked[i] = True
                    hids_masked[j] = True
                    self._object_assignments[o] = h

                    self._record_event('MATCH', o, dists[i, j])
                    self._object_last_matched[o] = frameid
                    self._hypothesis_last_seen[h] = frameid

            # 2. Try to remaining objects/hypotheses
            dists[oids_masked, :] = np.nan
            dists[:, hids_masked] = np.nan
            rids, cids = _linear_sum_assignment(dists)

            for i, j in zip(rids, cids):
                if not np.isfinite(dists[i, j]):
                    continue

                o = oids[i]
                h = hids[j]
                is_switch = (
                    o in self._object_assignments and
                    self._object_assignments[o] != h and
                    o in self._object_last_seen and
                    abs(frameid - self._object_last_seen[o]) <= self.max_switch_time
                )
                cat1 = 'SWITCH' if is_switch else 'MATCH'
                if cat1 == 'SWITCH':
                    if h not in self._hypothesis_last_seen:
                        subcat = 'ASCEND'
                        self._record_event(subcat, oids[i], dists[i, j])
                is_transfer = h in self._hypothesis_assignments and self._hypothesis_assignments[h] != o
                cat2 = 'TRANSFER' if is_transfer else 'MATCH'
                if cat2 == 'TRANSFER':
                    if o not in self._object_last_matched:
                        subcat = 'MIGRATE'
                        self._record_event(subcat, oids[i], dists[i, j])
                    self._record_event(cat2, oids[i], dists[i, j])
                self._hypothesis_last_seen[h] = frameid
                self._object_last_matched[o] = frameid
                self._record_event(cat1, oids[i], dists[i, j])
                oids_masked[i] = True
                hids_masked[j] = True
                self._object_assignments[o] = h
                self._hypothesis_assignments[h] = o

        # 3. All remaining objects are missed
        missed_oids = oids[~oids_masked]
        self._record_event_batch('MISS', len(missed_oids), oids=missed_oids)

        # 4. All remaining hypotheses are false alarms
        false_positive_hids = hids[~hids_masked]
        self._record_event_batch('FP', len(false_positive_hids))

        # 5. Update occurrence state
        for o in oids:
            self._object_last_seen[o] = frameid

        self._last_frame_id = frameid

        return frameid

    @property
    def metrics_engine(self):
        """Incremental statistics consumed by metric functions."""
        return self._metrics_engine
