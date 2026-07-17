# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Incremental statistics for the accumulator-only metric engine."""

from collections import defaultdict

import numpy as np

MATCH_EVENT_TYPES = ("SWITCH", "MATCH")


def _is_missing(value):
    """Return whether a scalar identity is missing."""
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


class _MetricsEngine(object):
    """Incremental primitive statistics and sparse identity associations."""

    def __init__(self):
        self.num_frames = 0
        self.type_counts = defaultdict(int)
        self.distance_sum = 0.0
        self.fragmentations = 0

        self.object_ids = []
        self.prediction_ids = []
        self._object_codes = {}
        self._prediction_codes = {}

        self.object_counts = []
        self.prediction_counts = []
        self.tracked_counts = []
        self._ever_tracked = []
        self._last_tracked = []

        self._raw_object_chunks = []
        self._raw_prediction_chunks = []
        self._raw_edge_cache = None

    def record_frame(self, oids, hids, valid_i, valid_j):
        """Record identities present in one frame and its finite raw edges."""
        self.num_frames += 1
        object_codes = np.fromiter(
            (self._object_code(oid) for oid in oids),
            dtype=np.intp,
            count=len(oids),
        )
        prediction_codes = np.fromiter(
            (self._prediction_code(hid) for hid in hids),
            dtype=np.intp,
            count=len(hids),
        )

        for code in object_codes:
            if code >= 0:
                self.object_counts[code] += 1
        for code in prediction_codes:
            if code >= 0:
                self.prediction_counts[code] += 1
        if len(valid_i):
            raw_object_codes = object_codes[valid_i]
            raw_prediction_codes = prediction_codes[valid_j]
            valid_codes = (raw_object_codes >= 0) & (raw_prediction_codes >= 0)
            if valid_codes.any():
                self._raw_object_chunks.append(raw_object_codes[valid_codes].copy())
                self._raw_prediction_chunks.append(raw_prediction_codes[valid_codes].copy())
        self._raw_edge_cache = None

    def record_event(self, event_type, oid, distance):
        """Record one derived (non-RAW) event."""
        if event_type == "RAW":
            return
        self.type_counts[event_type] += 1
        if event_type == "MISS":
            code = self._object_codes.get(oid, -1)
            if code >= 0:
                self._last_tracked[code] = False
            return
        if event_type not in MATCH_EVENT_TYPES:
            return

        object_code = self._object_codes.get(oid, -1)
        if object_code >= 0:
            self.tracked_counts[object_code] += 1
            if self._ever_tracked[object_code] and not self._last_tracked[object_code]:
                self.fragmentations += 1
            self._ever_tracked[object_code] = True
            self._last_tracked[object_code] = True
        if np.isfinite(distance):
            self.distance_sum += float(distance)

    def record_event_batch(self, event_type, count, oids=None):
        """Record a batch event without iterating for types that need only a count."""
        if event_type == "RAW" or count == 0:
            return
        self.type_counts[event_type] += int(count)
        if event_type == "MISS" and oids is not None:
            for oid in oids:
                code = self._object_codes.get(oid, -1)
                if code >= 0:
                    self._last_tracked[code] = False

    def object_frequencies(self):
        """Return compact object occurrence counts."""
        return np.asarray(self.object_counts, dtype=np.int64)

    def prediction_frequencies(self):
        """Return compact prediction occurrence counts."""
        return np.asarray(self.prediction_counts, dtype=np.int64)

    def track_ratios(self):
        """Return the tracked fraction for every ground-truth identity."""
        counts = np.asarray(self.object_counts, dtype=float)
        tracked = np.asarray(self.tracked_counts, dtype=float)
        ratios = np.divide(
            tracked,
            counts,
            out=np.zeros_like(tracked),
            where=counts != 0,
        )
        return ratios

    def raw_association_edges(self):
        """Return unique finite input edges as ``(object, prediction, count)``."""
        if self._raw_edge_cache is not None:
            return self._raw_edge_cache
        if not self._raw_object_chunks:
            self._raw_edge_cache = _empty_edges()
            return self._raw_edge_cache

        object_codes = np.concatenate(self._raw_object_chunks)
        prediction_codes = np.concatenate(self._raw_prediction_chunks)

        num_predictions = len(self.prediction_ids)
        flat_codes = object_codes.astype(np.int64) * num_predictions + prediction_codes
        unique_codes, counts = np.unique(flat_codes, return_counts=True)
        self._raw_edge_cache = (
            (unique_codes // num_predictions).astype(np.intp),
            (unique_codes % num_predictions).astype(np.intp),
            counts.astype(np.int64, copy=False),
        )
        return self._raw_edge_cache

    def _object_code(self, oid):
        if _is_missing(oid):
            return -1
        code = self._object_codes.get(oid)
        if code is None:
            code = len(self.object_ids)
            self._object_codes[oid] = code
            self.object_ids.append(oid)
            self.object_counts.append(0)
            self.tracked_counts.append(0)
            self._ever_tracked.append(False)
            self._last_tracked.append(False)
        return code

    def _prediction_code(self, hid):
        if _is_missing(hid):
            return -1
        code = self._prediction_codes.get(hid)
        if code is None:
            code = len(self.prediction_ids)
            self._prediction_codes[hid] = code
            self.prediction_ids.append(hid)
            self.prediction_counts.append(0)
        return code


def _empty_edges():
    empty = np.empty(0, dtype=np.intp)
    return empty, empty.copy(), np.empty(0, dtype=np.int64)
