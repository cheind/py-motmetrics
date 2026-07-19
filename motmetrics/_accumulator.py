"""Array-coded CLEAR matching for one prepared MOTChallenge sequence."""

import numpy as np

from motmetrics._assignment import _linear_sum_assignment
from motmetrics._metrics_engine import _MetricsEngine

_EMPTY_INTP = np.empty(0, dtype=np.intp)
_EMPTY_FLOAT = np.empty(0, dtype=float)
_EMPTY_BOOL = np.empty(0, dtype=np.bool_)


class _Accumulator(object):
    """Match prepared integer identity codes and update compact metrics."""

    def __init__(self, object_counts, prediction_counts, event_recorder=None):
        self._object_counts = np.asarray(object_counts, dtype=np.int64)
        self._prediction_counts = np.asarray(prediction_counts, dtype=np.int64)
        num_objects = len(self._object_counts)
        num_predictions = len(self._prediction_counts)
        self._object_assignments = np.full(num_objects, -1, dtype=np.intp)
        self._hypothesis_assignments = np.full(num_predictions, -1, dtype=np.intp)
        self._object_ever_matched = np.zeros(num_objects, dtype=np.bool_)
        self._hypothesis_ever_matched = np.zeros(num_predictions, dtype=np.bool_)
        self._matched_previous = np.zeros(num_objects, dtype=np.bool_)
        self._previous_matched_objects = np.empty(0, dtype=np.intp)
        self._local_hypothesis_columns = np.full(num_predictions, -1, dtype=np.intp)
        self._event_recorder = event_recorder
        self._metrics_engine = _MetricsEngine(
            self._object_counts,
            self._prediction_counts,
        )

    def update(
        self,
        object_codes,
        hypothesis_codes,
        distances,
        finite,
        frame_id=None,
    ):  # noqa: C901
        """Match one frame of globally coded identities."""
        distances = distances.reshape(
            len(object_codes),
            len(hypothesis_codes),
        )
        if self._event_recorder is not None:
            self._event_recorder.record_raw(
                frame_id,
                object_codes,
                hypothesis_codes,
                distances,
                finite,
            )
        object_mask = np.zeros(len(object_codes), dtype=np.bool_)
        hypothesis_mask = np.zeros(len(hypothesis_codes), dtype=np.bool_)
        advances_previous_frame = bool(
            len(object_codes) and len(hypothesis_codes)
        )

        finite_rows, finite_columns = np.nonzero(finite)
        self._metrics_engine.record_frame(
            object_codes[finite_rows],
            hypothesis_codes[finite_columns],
        )

        carried_rows = _EMPTY_INTP
        carried_columns = _EMPTY_INTP
        carried_distances = _EMPTY_FLOAT
        assigned_rows = _EMPTY_INTP
        assigned_columns = _EMPTY_INTP
        assigned_hypotheses = _EMPTY_INTP
        switch_mask = _EMPTY_BOOL
        transfer_mask = _EMPTY_BOOL
        ascend_mask = _EMPTY_BOOL
        migrate_mask = _EMPTY_BOOL
        ascend_count = 0
        migrate_count = 0

        if len(object_codes) and len(hypothesis_codes):
            local_columns = self._local_hypothesis_columns
            local_columns[hypothesis_codes] = np.arange(
                len(hypothesis_codes),
                dtype=np.intp,
            )

            candidate_rows = np.flatnonzero(self._matched_previous[object_codes])
            if len(candidate_rows):
                previous_hypotheses = self._object_assignments[
                    object_codes[candidate_rows]
                ]
                candidate_columns = local_columns[previous_hypotheses]
                present = candidate_columns >= 0
                candidate_rows = candidate_rows[present]
                candidate_columns = candidate_columns[present]
                if len(candidate_rows):
                    candidate_finite = finite[candidate_rows, candidate_columns]
                    carried_rows = candidate_rows[candidate_finite]
                    carried_columns = candidate_columns[candidate_finite]
                    carried_distances = distances[carried_rows, carried_columns]
                    object_mask[carried_rows] = True
                    hypothesis_mask[carried_columns] = True

            if len(carried_rows):
                finite[carried_rows, :] = False
                finite[:, carried_columns] = False
            assigned_rows, assigned_columns = _linear_sum_assignment(
                distances,
                finite=finite,
            )

            if len(assigned_rows):
                assigned_objects = object_codes[assigned_rows]
                assigned_hypotheses = hypothesis_codes[assigned_columns]
                previous_hypotheses = self._object_assignments[assigned_objects]
                previous_objects = self._hypothesis_assignments[assigned_hypotheses]
                switch_mask = (
                    (previous_hypotheses >= 0)
                    & (previous_hypotheses != assigned_hypotheses)
                )
                transfer_mask = (
                    (previous_objects >= 0)
                    & (previous_objects != assigned_objects)
                )
                ascend_mask = (
                    switch_mask
                    & ~self._hypothesis_ever_matched[assigned_hypotheses]
                )
                migrate_mask = (
                    transfer_mask
                    & ~self._object_ever_matched[assigned_objects]
                )
                ascend_count = int(np.count_nonzero(ascend_mask))
                migrate_count = int(np.count_nonzero(migrate_mask))

                self._object_assignments[assigned_objects] = assigned_hypotheses
                self._hypothesis_assignments[assigned_hypotheses] = assigned_objects
                self._object_ever_matched[assigned_objects] = True
                self._hypothesis_ever_matched[assigned_hypotheses] = True
                object_mask[assigned_rows] = True
                hypothesis_mask[assigned_columns] = True

            local_columns[hypothesis_codes] = -1

        carried_objects = object_codes[carried_rows]
        assigned_objects = object_codes[assigned_rows]
        assigned_distances = distances[assigned_rows, assigned_columns]
        matched_objects = np.concatenate((carried_objects, assigned_objects))
        matched_distances = np.concatenate(
            (
                carried_distances,
                assigned_distances,
            )
        )
        if self._event_recorder is not None:
            self._event_recorder.record_matches(
                frame_id,
                carried_objects,
                hypothesis_codes[carried_columns],
                carried_distances,
                assigned_objects,
                assigned_hypotheses,
                assigned_distances,
                switch_mask,
                transfer_mask,
                ascend_mask,
                migrate_mask,
            )
        self._metrics_engine.record_detections(
            matched_objects,
            matched_distances,
            num_switches=np.count_nonzero(switch_mask),
            num_transfer=np.count_nonzero(transfer_mask),
            num_ascend=ascend_count,
            num_migrate=migrate_count,
            advances_previous_frame=advances_previous_frame,
        )

        missed_objects = object_codes[~object_mask]
        self._metrics_engine.record_misses(missed_objects)
        if self._event_recorder is not None:
            false_positive_hypotheses = hypothesis_codes[~hypothesis_mask]
            self._metrics_engine.record_false_positives(
                len(false_positive_hypotheses)
            )
            self._event_recorder.record_misses(frame_id, missed_objects)
            self._event_recorder.record_false_positives(
                frame_id,
                false_positive_hypotheses,
            )
        else:
            self._metrics_engine.record_false_positives(
                np.count_nonzero(~hypothesis_mask)
            )

        if advances_previous_frame:
            self._matched_previous[self._previous_matched_objects] = False
            self._matched_previous[matched_objects] = True
            self._previous_matched_objects = matched_objects

    @property
    def metrics_engine(self):
        return self._metrics_engine
