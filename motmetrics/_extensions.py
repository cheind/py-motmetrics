# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Explicit, opt-in metric-family extension contract."""

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from types import MappingProxyType

import numpy as np

SUPPORTED_INTERMEDIATES = frozenset((
    "frame_iou",
    "trajectories",
    "clear_statistics",
    "clear_events",
))


class _ImmutableView(object):
    __slots__ = ()

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError("Metric input views are immutable.")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        del name
        raise AttributeError("Metric input views are immutable.")


class MetricFamily(object, metaclass=ABCMeta):
    """Base class for an explicitly supplied metric family.

    Subclasses declare their public output names and optional intermediate-data
    requirements, then implement sequence evaluation, sequence summarization,
    and cross-sequence combination. Instances are treated as immutable
    configuration and must be picklable when ``n_jobs > 1``.
    """

    name = None
    metric_names = ()
    requirements = frozenset()
    display_names = MappingProxyType({})
    formatters = MappingProxyType({})

    @abstractmethod
    def evaluate_sequence(self, sequence, intermediates):
        """Return compact, picklable state for one sequence."""
        raise NotImplementedError

    @abstractmethod
    def summarize(self, partial):
        """Return a mapping containing this family's per-sequence metrics."""
        raise NotImplementedError

    @abstractmethod
    def combine(self, partials):
        """Return a mapping containing this family's ``OVERALL`` metrics."""
        raise NotImplementedError


class DetectionView(_ImmutableView):
    """Immutable columnar detections supplied to custom metric families."""

    __slots__ = ("_frame_ids", "_ids", "_columns", "_field_names", "_boxes")

    def __init__(self, data):
        self._frame_ids = _read_only(data.frame_ids)
        self._ids = _read_only(data.ids)
        self._field_names = tuple(data.field_names)
        self._columns = {
            name: _read_only(data.column(name))
            for name in self._field_names
        }
        self._boxes = None

    def __len__(self):
        return len(self._frame_ids)

    @property
    def frame_ids(self):
        return self._frame_ids

    @property
    def ids(self):
        return self._ids

    @property
    def field_names(self):
        return self._field_names

    def column(self, name):
        """Return one immutable input column."""
        try:
            return self._columns[name]
        except KeyError as exc:
            raise KeyError("Unknown detection field: {!r}".format(name)) from exc

    def values(self, names):
        """Return an immutable matrix containing the requested columns."""
        names = tuple(names)
        if not names:
            return _read_only(np.empty((len(self), 0), dtype=float))
        return _read_only(np.column_stack([self.column(name) for name in names]))

    @property
    def boxes(self):
        """Return ``X, Y, Width, Height`` as an immutable matrix."""
        if self._boxes is None:
            object.__setattr__(
                self,
                "_boxes",
                self.values(("X", "Y", "Width", "Height")),
            )
        return self._boxes

    @property
    def confidence(self):
        """Return the input confidence column."""
        return self.column("Confidence")


class SequenceView(_ImmutableView):
    """Immutable raw inputs for one evaluated sequence."""

    __slots__ = ("name", "ground_truth", "tracker", "_frame_ids")

    def __init__(self, name, ground_truth, tracker):
        self.name = name
        self.ground_truth = DetectionView(ground_truth)
        self.tracker = DetectionView(tracker)
        self._frame_ids = _read_only(
            np.union1d(self.ground_truth.frame_ids, self.tracker.frame_ids)
        )

    @property
    def frame_ids(self):
        return self._frame_ids


class FrameIoU(_ImmutableView):
    """Immutable per-frame identities and IoU similarities."""

    __slots__ = (
        "frame_id",
        "ground_truth_ids",
        "tracker_ids",
        "similarities",
    )

    def __init__(self, frame_id, ground_truth_ids, tracker_ids, similarities):
        self.frame_id = frame_id
        self.ground_truth_ids = _read_only(ground_truth_ids)
        self.tracker_ids = _read_only(tracker_ids)
        self.similarities = _read_only(similarities)


class Trajectory(_ImmutableView):
    """Immutable view of all detections belonging to one identity."""

    __slots__ = ("identity", "_detections", "_indices", "_cache")

    def __init__(self, identity, detections, indices):
        self.identity = identity
        self._detections = detections
        self._indices = _read_only(indices)
        self._cache = {}

    def __len__(self):
        return len(self._indices)

    @property
    def frame_ids(self):
        return self._column("__frame_ids__", self._detections.frame_ids)

    @property
    def boxes(self):
        return self._column("__boxes__", self._detections.boxes)

    @property
    def confidence(self):
        return self.column("Confidence")

    def column(self, name):
        """Return one immutable field for this trajectory."""
        return self._column(name, self._detections.column(name))

    def _column(self, name, values):
        if name not in self._cache:
            self._cache[name] = _read_only(values[self._indices])
        return self._cache[name]


class TrajectorySet(_ImmutableView):
    """Ground-truth and tracker trajectories for one sequence."""

    __slots__ = ("ground_truth", "tracker")

    def __init__(self, ground_truth, tracker):
        self.ground_truth = ground_truth
        self.tracker = tracker


class ClearStatistics(_ImmutableView):
    """Immutable compact statistics already maintained by CLEAR matching."""

    __slots__ = (
        "ground_truth_ids",
        "ground_truth_detection_counts",
        "matched_ground_truth_detection_counts",
        "tracker_ids",
        "tracker_detection_counts",
        "event_counts",
        "num_frames",
        "distance_sum",
        "fragmentations",
        "_track_coverage",
    )

    def __init__(self, prepared):
        engine = prepared.accumulator.metrics_engine
        self.ground_truth_ids = _read_only(prepared.ground_truth_ids)
        self.ground_truth_detection_counts = _read_only(engine.object_counts)
        self.matched_ground_truth_detection_counts = _read_only(
            engine.tracked_counts
        )
        self.tracker_ids = _read_only(prepared.tracker_ids)
        self.tracker_detection_counts = _read_only(engine.prediction_counts)
        self.event_counts = MappingProxyType(dict(engine.type_counts))
        self.num_frames = engine.num_frames
        self.distance_sum = engine.distance_sum
        self.fragmentations = engine.fragmentations
        self._track_coverage = None

    @property
    def track_coverage(self):
        """Return each GT trajectory's matched fraction of its lifespan."""
        if self._track_coverage is None:
            ratios = np.divide(
                self.matched_ground_truth_detection_counts,
                self.ground_truth_detection_counts,
                out=np.zeros(
                    len(self.ground_truth_detection_counts),
                    dtype=float,
                ),
                where=self.ground_truth_detection_counts != 0,
            )
            object.__setattr__(self, "_track_coverage", _read_only(ratios))
        return self._track_coverage


class ClearEventLog(_ImmutableView):
    """Immutable, opt-in CLEAR event history with a lazy pandas view."""

    __slots__ = (
        "frame_ids",
        "event_ids",
        "types",
        "ground_truth_codes",
        "tracker_codes",
        "distances",
        "ground_truth_id_values",
        "tracker_id_values",
        "_ground_truth_ids",
        "_tracker_ids",
        "_df",
    )

    def __init__(
        self,
        frame_ids,
        event_ids,
        types,
        ground_truth_codes,
        tracker_codes,
        distances,
        ground_truth_id_values,
        tracker_id_values,
    ):
        self.frame_ids = _read_only(frame_ids)
        self.event_ids = _read_only(event_ids)
        self.types = _read_only(types)
        self.ground_truth_codes = _read_only(ground_truth_codes)
        self.tracker_codes = _read_only(tracker_codes)
        self.distances = _read_only(distances)
        self.ground_truth_id_values = _read_only(ground_truth_id_values)
        self.tracker_id_values = _read_only(tracker_id_values)
        self._ground_truth_ids = None
        self._tracker_ids = None
        self._df = None

    def __len__(self):
        return len(self.frame_ids)

    @property
    def ground_truth_ids(self):
        if self._ground_truth_ids is None:
            object.__setattr__(
                self,
                "_ground_truth_ids",
                _expand_event_ids(
                    self.ground_truth_codes,
                    self.ground_truth_id_values,
                ),
            )
        return self._ground_truth_ids

    @property
    def tracker_ids(self):
        if self._tracker_ids is None:
            object.__setattr__(
                self,
                "_tracker_ids",
                _expand_event_ids(
                    self.tracker_codes,
                    self.tracker_id_values,
                ),
            )
        return self._tracker_ids

    @property
    def df(self):
        """Materialize a pandas event DataFrame only when explicitly requested."""
        if self._df is None:
            import pandas as pd

            index = pd.MultiIndex.from_arrays(
                (self.frame_ids, self.event_ids),
                names=("FrameId", "Event"),
            )
            object.__setattr__(
                self,
                "_df",
                pd.DataFrame(
                    OrderedDict((
                        ("Type", self.types),
                        ("OId", self.ground_truth_ids),
                        ("HId", self.tracker_ids),
                        ("D", self.distances),
                    )),
                    index=index,
                ),
            )
        return self._df


class MetricContext(object):
    """Declared, lazy intermediates supplied to one custom metric family."""

    __slots__ = ("sequence", "_store", "_requirements")

    def __init__(
        self,
        sequence,
        prepared,
        clear_events=None,
        requirements=(),
    ):
        self.sequence = sequence
        self._store = _IntermediateStore(sequence, prepared, clear_events)
        self._requirements = frozenset(requirements)

    @classmethod
    def _from_store(cls, store, requirements):
        context = cls.__new__(cls)
        context.sequence = store.sequence
        context._store = store
        context._requirements = frozenset(requirements)
        return context

    def _for_requirements(self, requirements):
        return self._from_store(self._store, requirements)

    @property
    def frame_iou(self):
        """Return immutable per-frame IoU data, materialized on first access."""
        self._require("frame_iou")
        if self._store.frame_iou is None:
            prepared = self._store.prepared
            self._store.frame_iou = tuple(
                FrameIoU(
                    frame_id,
                    prepared.ground_truth_ids[ground_truth_indices],
                    prepared.tracker_ids[tracker_indices],
                    similarities,
                )
                for frame_id, (
                    ground_truth_indices,
                    tracker_indices,
                    similarities,
                ) in zip(prepared.frame_ids, prepared.frame_data)
            )
        return self._store.frame_iou

    @property
    def trajectories(self):
        """Return identity-grouped detections, materialized on first access."""
        self._require("trajectories")
        if self._store.trajectories is None:
            self._store.trajectories = TrajectorySet(
                _build_trajectories(self.sequence.ground_truth),
                _build_trajectories(self.sequence.tracker),
            )
        return self._store.trajectories

    @property
    def clear_events(self):
        """Return the opt-in compact CLEAR event history."""
        self._require("clear_events")
        return self._store.clear_events

    @property
    def clear_statistics(self):
        """Return compact counts already produced by CLEAR matching."""
        self._require("clear_statistics")
        if self._store.clear_statistics is None:
            self._store.clear_statistics = ClearStatistics(
                self._store.prepared
            )
        return self._store.clear_statistics

    def _require(self, requirement):
        if requirement not in self._requirements:
            raise RuntimeError(
                "This family must declare the {!r} requirement.".format(
                    requirement
                )
            )


class _IntermediateStore(object):
    """Share lazy provider results without widening a family's permissions."""

    __slots__ = (
        "sequence",
        "prepared",
        "clear_events",
        "clear_statistics",
        "frame_iou",
        "trajectories",
    )

    def __init__(self, sequence, prepared, clear_events):
        self.sequence = sequence
        self.prepared = prepared
        self.clear_events = clear_events
        self.clear_statistics = None
        self.frame_iou = None
        self.trajectories = None


class _ClearEventRecorder(object):
    """Collect old-style CLEAR events only for families that request them."""

    __slots__ = (
        "_frame_ids",
        "_event_ids",
        "_types",
        "_ground_truth_codes",
        "_tracker_codes",
        "_distances",
        "_ground_truth_id_values",
        "_tracker_id_values",
        "_last_frame",
        "_next_event_id",
        "_log",
    )

    def __init__(self, ground_truth_id_values, tracker_id_values):
        self._frame_ids = []
        self._event_ids = []
        self._types = []
        self._ground_truth_codes = []
        self._tracker_codes = []
        self._distances = []
        self._ground_truth_id_values = np.asarray(ground_truth_id_values)
        self._tracker_id_values = np.asarray(tracker_id_values)
        self._last_frame = None
        self._next_event_id = 0
        self._log = None

    def record_raw(self, frame_id, object_codes, hypothesis_codes, distances, finite):
        self._append(frame_id, "RAW")
        rows, columns = np.nonzero(finite)
        for row, column in zip(rows, columns):
            self._append(
                frame_id,
                "RAW",
                object_codes[row],
                hypothesis_codes[column],
                distances[row, column],
            )
        used_rows = np.zeros(len(object_codes), dtype=bool)
        used_columns = np.zeros(len(hypothesis_codes), dtype=bool)
        used_rows[rows] = True
        used_columns[columns] = True
        for object_code in object_codes[~used_rows]:
            self._append(frame_id, "RAW", ground_truth_code=object_code)
        for hypothesis_code in hypothesis_codes[~used_columns]:
            self._append(frame_id, "RAW", tracker_code=hypothesis_code)

    def record_matches(
        self,
        frame_id,
        carried_objects,
        carried_hypotheses,
        carried_distances,
        assigned_objects,
        assigned_hypotheses,
        assigned_distances,
        switch_mask,
        transfer_mask,
        ascend_mask,
        migrate_mask,
    ):
        for object_code, hypothesis_code, distance in zip(
            carried_objects,
            carried_hypotheses,
            carried_distances,
        ):
            self._append(
                frame_id,
                "MATCH",
                object_code,
                hypothesis_code,
                distance,
            )
        for index, (object_code, hypothesis_code, distance) in enumerate(zip(
            assigned_objects,
            assigned_hypotheses,
            assigned_distances,
        )):
            if ascend_mask[index]:
                self._append(frame_id, "ASCEND", object_code, hypothesis_code, distance)
            if migrate_mask[index]:
                self._append(frame_id, "MIGRATE", object_code, hypothesis_code, distance)
            if transfer_mask[index]:
                self._append(frame_id, "TRANSFER", object_code, hypothesis_code, distance)
            event_type = "SWITCH" if switch_mask[index] else "MATCH"
            self._append(frame_id, event_type, object_code, hypothesis_code, distance)

    def record_misses(self, frame_id, object_codes):
        for object_code in object_codes:
            self._append(frame_id, "MISS", ground_truth_code=object_code)

    def record_false_positives(self, frame_id, hypothesis_codes):
        for hypothesis_code in hypothesis_codes:
            self._append(frame_id, "FP", tracker_code=hypothesis_code)

    def finish(self):
        if self._log is None:
            self._log = ClearEventLog(
                np.asarray(self._frame_ids, dtype=np.int64),
                np.asarray(self._event_ids, dtype=np.int64),
                np.asarray(self._types, dtype="U8"),
                np.asarray(self._ground_truth_codes, dtype=np.intp),
                np.asarray(self._tracker_codes, dtype=np.intp),
                np.asarray(self._distances, dtype=float),
                self._ground_truth_id_values,
                self._tracker_id_values,
            )
        return self._log

    def _append(
        self,
        frame_id,
        event_type,
        ground_truth_code=-1,
        tracker_code=-1,
        distance=np.nan,
    ):
        if frame_id != self._last_frame:
            self._last_frame = frame_id
            self._next_event_id = 0
        self._frame_ids.append(frame_id)
        self._event_ids.append(self._next_event_id)
        self._types.append(event_type)
        self._ground_truth_codes.append(int(ground_truth_code))
        self._tracker_codes.append(int(tracker_code))
        self._distances.append(float(distance))
        self._next_event_id += 1


def _build_trajectories(detections):
    if len(detections) == 0:
        return ()
    order = np.argsort(detections.ids, kind="stable")
    sorted_ids = detections.ids[order]
    boundaries = np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1
    return tuple(
        Trajectory(sorted_ids[group[0]], detections, group)
        for group in np.split(order, boundaries)
    )


def _expand_event_ids(codes, id_values):
    values = np.empty(len(codes), dtype=object)
    values[:] = None
    present = codes >= 0
    values[present] = id_values[codes[present]]
    return _read_only(values)


def _read_only(values):
    values = np.asarray(values).view()
    values.flags.writeable = False
    return values
