# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Compute metrics from incremental accumulator state."""

# pylint: disable=redefined-outer-name

import numpy as np

from motmetrics._accumulator import _Accumulator
from motmetrics._assignment import _linear_sum_assignment


def _compute_metrics(accumulator, metric_names=None):
    """Compute a fixed set of metrics from compact sequence state."""
    if not isinstance(accumulator, _Accumulator):
        raise TypeError("The metric engine requires an accumulator.")
    metric_names = _normalize_metric_names(metric_names)
    cache = {}
    for metric_name in metric_names:
        _resolve_metric(accumulator.metrics_engine, metric_name, cache)
    return cache


def _compute_overall(partials, metric_names=None):
    """Merge fixed per-sequence metric state into an overall result."""
    metric_names = _normalize_metric_names(metric_names)
    cache = {}
    for metric_name in metric_names:
        _resolve_overall_metric(partials, metric_name, cache)
    return cache


def _normalize_metric_names(metric_names):
    if metric_names is None:
        return _MOTCHALLENGE_METRICS
    if isinstance(metric_names, str):
        return (metric_names,)
    return tuple(metric_names)


def _resolve_metric(engine, name, cache):
    if name in cache:
        return cache[name]
    try:
        function, dependencies, _, _ = _METRIC_SPECS[name]
    except KeyError as exc:
        raise ValueError("Unknown metric: {}".format(name)) from exc
    values = [_resolve_metric(engine, dependency, cache) for dependency in dependencies]
    cache[name] = function(engine, *values)
    return cache[name]


def _resolve_overall_metric(partials, name, cache):
    if name in cache:
        return cache[name]
    try:
        _, _, merge, dependencies = _METRIC_SPECS[name]
    except KeyError as exc:
        raise ValueError("Unknown metric: {}".format(name)) from exc
    if merge is None:
        raise ValueError("Metric cannot be combined across sequences: {}".format(name))
    values = [
        _resolve_overall_metric(partials, dependency, cache)
        for dependency in dependencies
    ]
    cache[name] = merge(partials, *values)
    return cache[name]


def _required_arguments(function):
    """Return positional argument names without importing ``inspect``."""
    code = function.__code__
    optional_count = len(function.__defaults__ or ())
    return code.co_varnames[: code.co_argcount - optional_count]


def num_frames(engine):
    """Total number of frames."""
    return engine.num_frames


def obj_frequencies(engine):
    """Total number of occurrences of individual objects over all frames."""
    return engine.object_frequencies()


def pred_frequencies(engine):
    """Total number of occurrences of individual predictions over all frames."""
    return engine.prediction_frequencies()


def num_unique_objects(engine, obj_frequencies):
    """Total number of unique object ids encountered."""
    del engine  # unused
    return len(obj_frequencies)


def num_matches(engine):
    """Total number matches."""
    return engine.type_counts["MATCH"]


def num_switches(engine):
    """Total number of track switches."""
    return engine.type_counts["SWITCH"]


def num_transfer(engine):
    """Total number of track transfer."""
    return engine.type_counts["TRANSFER"]


def num_ascend(engine):
    """Total number of track ascend."""
    return engine.type_counts["ASCEND"]


def num_migrate(engine):
    """Total number of track migrate."""
    return engine.type_counts["MIGRATE"]


def num_false_positives(engine):
    """Total number of false positives (false-alarms)."""
    return engine.type_counts["FP"]


def num_misses(engine):
    """Total number of misses."""
    return engine.type_counts["MISS"]


def num_detections(engine, num_matches, num_switches):
    """Total number of detected objects including matches and switches."""
    del engine  # unused
    return num_matches + num_switches


def num_objects(engine, obj_frequencies):
    """Total number of unique object appearances over all frames."""
    del engine  # unused
    return obj_frequencies.sum()


def num_predictions(engine, pred_frequencies):
    """Total number of unique prediction appearances over all frames."""
    del engine  # unused
    return pred_frequencies.sum()


def num_gt_ids(engine):
    """Number of unique gt ids."""
    return len(engine.object_ids)


def num_dt_ids(engine):
    """Number of unique dt ids."""
    return len(engine.prediction_ids)


def track_ratios(engine, obj_frequencies):
    """Ratio of assigned to total appearance count per unique object id."""
    del obj_frequencies  # already represented by the compact engine
    return engine.track_ratios()


def mostly_tracked(engine, track_ratios):
    """Number of objects tracked for more than 80 percent of lifespan."""
    del engine  # unused
    return int(np.count_nonzero(track_ratios > 0.8))


def partially_tracked(engine, track_ratios):
    """Number of objects tracked between 20 and 80 percent of lifespan, inclusive."""
    del engine  # unused
    return int(np.count_nonzero((track_ratios >= 0.2) & (track_ratios <= 0.8)))


def mostly_lost(engine, track_ratios):
    """Number of objects tracked less than 20 percent of lifespan."""
    del engine  # unused
    return int(np.count_nonzero(track_ratios < 0.2))


def num_fragmentations(engine, obj_frequencies):
    """Total number of switches from tracked to not tracked."""
    del obj_frequencies  # unused
    return engine.fragmentations


def motp(engine, num_detections):
    """Multiple object tracker precision."""
    return _quiet_divide(engine.distance_sum, num_detections)


def motp_sum(engine, num_detections):
    """Sum of CLEAR match similarities used by TrackEval derivatives."""
    return num_detections - engine.distance_sum


def _merge_motp(partials, num_detections):
    res = 0
    for v in partials:
        res += v["motp"] * v["num_detections"]
    return _quiet_divide(res, num_detections)


def mota(engine, num_misses, num_switches, num_false_positives, num_objects):
    """Multiple object tracker accuracy."""
    del engine  # unused
    return 1.0 - _quiet_divide(
        num_misses + num_switches + num_false_positives, num_objects
    )


def moda(engine, num_detections, num_false_positives, num_objects):
    """Multiple object detection accuracy."""
    del engine  # unused
    return (num_detections - num_false_positives) / np.maximum(1.0, num_objects)


def smota(engine, motp_sum, num_false_positives, num_switches, num_objects):
    """Soft multiple object tracker accuracy."""
    del engine  # unused
    return (motp_sum - num_false_positives - num_switches) / np.maximum(1.0, num_objects)


def mtr(engine, mostly_tracked, partially_tracked, mostly_lost):
    """Fraction of ground-truth tracks that are mostly tracked."""
    del engine  # unused
    return mostly_tracked / np.maximum(1.0, mostly_tracked + partially_tracked + mostly_lost)


def ptr(engine, mostly_tracked, partially_tracked, mostly_lost):
    """Fraction of ground-truth tracks that are partially tracked."""
    del engine  # unused
    return partially_tracked / np.maximum(1.0, mostly_tracked + partially_tracked + mostly_lost)


def mlr(engine, mostly_tracked, partially_tracked, mostly_lost):
    """Fraction of ground-truth tracks that are mostly lost."""
    del engine  # unused
    return mostly_lost / np.maximum(1.0, mostly_tracked + partially_tracked + mostly_lost)


def clr_f1(engine, num_detections, num_misses, num_false_positives):
    """CLEAR detection F1 score."""
    del engine  # unused
    denominator = num_detections + 0.5 * num_misses + 0.5 * num_false_positives
    return num_detections / np.maximum(1.0, denominator)


def fp_per_frame(engine, num_false_positives, num_frames):
    """Average number of false positives per frame."""
    del engine  # unused
    return num_false_positives / np.maximum(1.0, num_frames)


def precision(engine, num_detections, num_false_positives):
    """Number of detected objects over sum of detected and false positives."""
    del engine  # unused
    return _quiet_divide(num_detections, num_false_positives + num_detections)


def recall(engine, num_detections, num_objects):
    """Number of detections over number of objects."""
    del engine  # unused
    return _quiet_divide(num_detections, num_objects)


def id_global_assignment(engine):
    """ID measures: maximum-overlap assignment on sparse connected components."""
    object_codes, prediction_codes, counts = engine.raw_association_edges()
    rids, cids, idtp_value = _max_weight_matching(
        len(engine.object_ids),
        len(engine.prediction_ids),
        object_codes,
        prediction_codes,
        counts,
    )
    num_objects = int(np.sum(engine.object_counts))
    num_predictions = int(np.sum(engine.prediction_counts))
    idfn_value = num_objects - idtp_value
    idfp_value = num_predictions - idtp_value

    return {
        "rids": rids,
        "cids": cids,
        "idtp": idtp_value,
        "idfp": idfp_value,
        "idfn": idfn_value,
        "min_cost": idfp_value + idfn_value,
    }


def _max_weight_matching(num_objects, num_predictions, object_codes, prediction_codes, weights):
    """Solve independent positive-weight bipartite components without global padding."""
    if not len(weights):
        empty = np.empty(0, dtype=int)
        return empty, empty.copy(), 0

    # A compact rectangular solve is fastest while its memory remains modest.
    # Above this boundary, decompose the sparse graph so unrelated identities
    # never inflate one global dense matrix.
    if num_objects * num_predictions <= 20_000_000:
        cost_dtype = np.int32 if int(weights.max()) <= np.iinfo(np.int32).max else np.int64
        costs = np.zeros((num_objects, num_predictions), dtype=cost_dtype)
        costs[object_codes, prediction_codes] = -weights
        rids, cids = _linear_sum_assignment(costs)
        positive = costs[rids, cids] < 0
        rids = rids[positive]
        cids = cids[positive]
        return rids, cids, int(-costs[rids, cids].sum(dtype=np.int64))

    edge_components = _bipartite_edge_components(
        num_objects,
        num_predictions,
        object_codes,
        prediction_codes,
    )
    order = np.argsort(edge_components, kind="stable")
    boundaries = np.flatnonzero(edge_components[order][1:] != edge_components[order][:-1]) + 1

    selected_objects = []
    selected_predictions = []
    true_positives = 0
    for component_edge_indices in np.split(order, boundaries):
        component_objects = np.unique(object_codes[component_edge_indices])
        component_predictions = np.unique(prediction_codes[component_edge_indices])
        if len(component_edge_indices) == 1:
            edge_index = component_edge_indices[0]
            selected_objects.append(object_codes[edge_index : edge_index + 1])
            selected_predictions.append(prediction_codes[edge_index : edge_index + 1])
            true_positives += int(weights[edge_index])
            continue

        component_costs = np.zeros(
            (len(component_objects), len(component_predictions)), dtype=np.int64
        )
        local_objects = np.searchsorted(
            component_objects, object_codes[component_edge_indices]
        )
        local_predictions = np.searchsorted(
            component_predictions, prediction_codes[component_edge_indices]
        )
        component_costs[local_objects, local_predictions] = -weights[component_edge_indices]
        local_rows, local_cols = _linear_sum_assignment(component_costs)
        positive = component_costs[local_rows, local_cols] < 0
        local_rows = local_rows[positive]
        local_cols = local_cols[positive]
        selected_objects.append(component_objects[local_rows])
        selected_predictions.append(component_predictions[local_cols])
        true_positives -= int(component_costs[local_rows, local_cols].sum())

    return (
        np.concatenate(selected_objects).astype(int, copy=False),
        np.concatenate(selected_predictions).astype(int, copy=False),
        true_positives,
    )


def _bipartite_edge_components(
    num_objects,
    num_predictions,
    object_codes,
    prediction_codes,
):
    """Label sparse bipartite edges with a compact union-find."""
    parent = np.arange(num_objects + num_predictions, dtype=np.intp)
    sizes = np.ones(len(parent), dtype=np.intp)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    prediction_offset = num_objects
    for object_code, prediction_code in zip(object_codes, prediction_codes):
        object_root = find(int(object_code))
        prediction_root = find(prediction_offset + int(prediction_code))
        if object_root == prediction_root:
            continue
        if sizes[object_root] < sizes[prediction_root]:
            object_root, prediction_root = prediction_root, object_root
        parent[prediction_root] = object_root
        sizes[object_root] += sizes[prediction_root]

    return np.fromiter(
        (find(int(object_code)) for object_code in object_codes),
        dtype=np.intp,
        count=len(object_codes),
    )


def idfp(engine, id_global_assignment):
    """ID measures: Number of false positive matches after global min-cost matching."""
    del engine  # unused
    return id_global_assignment["idfp"]


def idfn(engine, id_global_assignment):
    """ID measures: Number of false negatives matches after global min-cost matching."""
    del engine  # unused
    return id_global_assignment["idfn"]


def idtp(engine, id_global_assignment, num_objects, idfn):
    """ID measures: Number of true positives matches after global min-cost matching."""
    del engine, num_objects, idfn  # unused
    return id_global_assignment["idtp"]


def idp(engine, idtp, idfp):
    """ID measures: global min-cost precision."""
    del engine  # unused
    return _quiet_divide(idtp, idtp + idfp)


def idr(engine, idtp, idfn):
    """ID measures: global min-cost recall."""
    del engine  # unused
    return _quiet_divide(idtp, idtp + idfn)


def idf1(engine, idtp, num_objects, num_predictions):
    """ID measures: global min-cost F1 score."""
    del engine  # unused
    return _quiet_divide(2 * idtp, num_objects + num_predictions)


_METRIC_FUNCTIONS = (
    num_frames,
    obj_frequencies,
    pred_frequencies,
    num_matches,
    num_switches,
    num_transfer,
    num_ascend,
    num_migrate,
    num_false_positives,
    num_misses,
    num_detections,
    num_objects,
    num_predictions,
    num_gt_ids,
    num_dt_ids,
    num_unique_objects,
    track_ratios,
    mostly_tracked,
    partially_tracked,
    mostly_lost,
    num_fragmentations,
    motp,
    motp_sum,
    mota,
    moda,
    smota,
    mtr,
    ptr,
    mlr,
    clr_f1,
    fp_per_frame,
    precision,
    recall,
    id_global_assignment,
    idfp,
    idfn,
    idtp,
    idp,
    idr,
    idf1,
)

_ADDITIVE_METRICS = {
    num_frames,
    num_unique_objects,
    num_matches,
    num_switches,
    num_transfer,
    num_ascend,
    num_migrate,
    num_false_positives,
    num_misses,
    num_detections,
    num_objects,
    num_predictions,
    num_gt_ids,
    num_dt_ids,
    mostly_tracked,
    partially_tracked,
    mostly_lost,
    num_fragmentations,
    motp_sum,
    idfp,
    idfn,
    idtp,
}
_SAME_OVERALL_FORMULA = {
    mota,
    moda,
    smota,
    mtr,
    ptr,
    mlr,
    clr_f1,
    fp_per_frame,
    precision,
    recall,
    idp,
    idr,
    idf1,
}
_FORMATTERS = {
    "num_frames": "{:d}".format,
    "obj_frequencies": "{:d}".format,
    "pred_frequencies": "{:d}".format,
    "num_matches": "{:d}".format,
    "num_switches": "{:d}".format,
    "num_transfer": "{:d}".format,
    "num_ascend": "{:d}".format,
    "num_migrate": "{:d}".format,
    "num_false_positives": "{:d}".format,
    "num_misses": "{:d}".format,
    "num_detections": "{:d}".format,
    "num_objects": "{:d}".format,
    "num_predictions": "{:d}".format,
    "num_gt_ids": "{:d}".format,
    "num_dt_ids": "{:d}".format,
    "num_unique_objects": "{:d}".format,
    "mostly_tracked": "{:d}".format,
    "partially_tracked": "{:d}".format,
    "mostly_lost": "{:d}".format,
    "motp": "{:.3f}".format,
    "mota": "{:.1%}".format,
    "moda": "{:.1%}".format,
    "smota": "{:.1%}".format,
    "mtr": "{:.1%}".format,
    "ptr": "{:.1%}".format,
    "mlr": "{:.1%}".format,
    "clr_f1": "{:.1%}".format,
    "fp_per_frame": "{:.3f}".format,
    "precision": "{:.1%}".format,
    "recall": "{:.1%}".format,
    "idp": "{:.1%}".format,
    "idr": "{:.1%}".format,
    "idf1": "{:.1%}".format,
}


def _sum_partial(metric):
    metric_name = metric.__name__

    def merge(partials):
        return sum(partial[metric_name] for partial in partials)

    return merge


def _merge_same_formula(metric):
    def merge(partials, *values):
        del partials
        return metric(None, *values)

    return merge


def _build_metric_specs():
    specs = {}
    for metric in _METRIC_FUNCTIONS:
        dependencies = tuple(_required_arguments(metric)[1:])
        if metric in _ADDITIVE_METRICS:
            merge = _sum_partial(metric)
            overall_dependencies = ()
        elif metric is motp:
            merge = _merge_motp
            overall_dependencies = ("num_detections",)
        elif metric in _SAME_OVERALL_FORMULA:
            merge = _merge_same_formula(metric)
            overall_dependencies = dependencies
        else:
            merge = None
            overall_dependencies = ()
        specs[metric.__name__] = (
            metric,
            dependencies,
            merge,
            overall_dependencies,
        )
    return specs


def _quiet_divide(numerator, denominator):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.true_divide(numerator, denominator)


_MOTCHALLENGE_METRICS = (
    "idf1",
    "idp",
    "idr",
    "recall",
    "precision",
    "num_unique_objects",
    "mostly_tracked",
    "partially_tracked",
    "mostly_lost",
    "mtr",
    "ptr",
    "mlr",
    "num_false_positives",
    "num_misses",
    "num_switches",
    "num_fragmentations",
    "mota",
    "moda",
    "motp",
    "smota",
    "clr_f1",
    "fp_per_frame",
    "num_transfer",
    "num_ascend",
    "num_migrate",
)

_METRIC_SPECS = _build_metric_specs()
