"""Compact array-backed statistics for the MOTChallenge metric engine."""

from collections import defaultdict

import numpy as np


class _MetricsEngine(object):
    """Primitive counts and sparse identity edges for one prepared sequence."""

    def __init__(self, object_counts, prediction_counts):
        self.num_frames = 0
        self.type_counts = defaultdict(int)
        self.distance_sum = 0.0
        self.fragmentations = 0

        self.object_counts = np.asarray(object_counts, dtype=np.int64)
        self.prediction_counts = np.asarray(prediction_counts, dtype=np.int64)
        self.object_ids = np.arange(len(self.object_counts), dtype=np.intp)
        self.prediction_ids = np.arange(len(self.prediction_counts), dtype=np.intp)

        self.tracked_counts = np.zeros(len(self.object_counts), dtype=np.int64)
        self._ever_tracked = np.zeros(len(self.object_counts), dtype=np.bool_)
        self._last_tracked = np.zeros(len(self.object_counts), dtype=np.bool_)

        self._raw_edge_chunks = []
        self._raw_edge_cache = None

    def record_frame(self, object_codes, prediction_codes):
        """Record one frame and its finite thresholded identity edges."""
        self.num_frames += 1
        if len(object_codes):
            flat_codes = (
                object_codes * len(self.prediction_counts)
                + prediction_codes
            )
            self._raw_edge_chunks.append(flat_codes)
            self._raw_edge_cache = None

    def record_detections(
        self,
        object_codes,
        distances,
        num_switches=0,
        num_transfer=0,
        num_ascend=0,
        num_migrate=0,
    ):
        """Batch all matched-object state changes for one frame."""
        count = len(object_codes)
        if count == 0:
            return

        num_switches = int(num_switches)
        self.type_counts["MATCH"] += count - num_switches
        self.type_counts["SWITCH"] += num_switches
        self.type_counts["TRANSFER"] += int(num_transfer)
        self.type_counts["ASCEND"] += int(num_ascend)
        self.type_counts["MIGRATE"] += int(num_migrate)

        self.tracked_counts[object_codes] += 1
        self.fragmentations += int(
            np.count_nonzero(
                self._ever_tracked[object_codes]
                & ~self._last_tracked[object_codes]
            )
        )
        self._ever_tracked[object_codes] = True
        self._last_tracked[object_codes] = True
        self.distance_sum += float(np.sum(distances))

    def record_misses(self, object_codes):
        """Batch missed-object state changes for one frame."""
        count = len(object_codes)
        if count == 0:
            return
        self.type_counts["MISS"] += count
        self._last_tracked[object_codes] = False

    def record_false_positives(self, count):
        """Record false positives for one frame."""
        if count:
            self.type_counts["FP"] += int(count)

    def object_frequencies(self):
        return self.object_counts

    def prediction_frequencies(self):
        return self.prediction_counts

    def track_ratios(self):
        return np.divide(
            self.tracked_counts,
            self.object_counts,
            out=np.zeros(len(self.tracked_counts), dtype=float),
            where=self.object_counts != 0,
        )

    def raw_association_edges(self):
        """Return unique finite input edges as ``(object, prediction, count)``."""
        if self._raw_edge_cache is not None:
            return self._raw_edge_cache
        if not self._raw_edge_chunks:
            self._raw_edge_cache = _empty_edges()
            return self._raw_edge_cache

        unique_codes, counts = np.unique(
            np.concatenate(self._raw_edge_chunks),
            return_counts=True,
        )
        num_predictions = len(self.prediction_counts)
        self._raw_edge_cache = (
            (unique_codes // num_predictions).astype(np.intp),
            (unique_codes % num_predictions).astype(np.intp),
            counts.astype(np.int64, copy=False),
        )
        return self._raw_edge_cache


def _empty_edges():
    empty = np.empty(0, dtype=np.intp)
    return empty, empty.copy(), np.empty(0, dtype=np.int64)
