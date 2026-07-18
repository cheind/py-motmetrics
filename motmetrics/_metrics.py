# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Compute metrics from incremental accumulator state."""

# pylint: disable=redefined-outer-name

from collections import OrderedDict

import numpy as np

import motmetrics._math_util as math_util
from motmetrics._accumulator import _Accumulator
from motmetrics._assignment import _linear_sum_assignment


class _MetricsHost:
    """Keeps track of metrics and intra metric dependencies."""

    def __init__(self):
        self.metrics = OrderedDict()

    def _register(  # noqa: C901
        self,
        fnc,
        deps="auto",
        name=None,
        formatter=None,
        fnc_m=None,
        deps_m="auto",
    ):
        """Register a new metric.

        Params
        ------
        fnc : Function
            Function that computes the metric to be registered. The number of arguments
            is 1 + N, where N is the number of dependencies of the metric to be registered.
            The order of the argument passed is `engine, result_dep1, result_dep2, ...`.

        Kwargs
        ------
        deps : string, list of strings or None, optional
            The dependencies of this metric. Each dependency is evaluated and the result
            is passed as argument to `fnc` as described above. If None is specified, the
            function does not have any dependencies. If a list of strings is given, dependencies
            for these metric strings are registered. If 'auto' is passed, the dependencies
            are deduced from argument inspection of the method. For this to work the argument
            names have to be equal to the intended dependencies.
        name : string or None, optional
            Name identifier of this metric. If None is passed the name is deduced from
            function inspection.
        formatter: Format object, optional
            An optional default formatter when rendering metric results as string. I.e to
            render the result `0.35` as `35%` one would pass `{:.2%}.format`
        fnc_m : Function or None, optional
            Function that merges metric results. The number of arguments
            is 1 + N, where N is the number of dependencies of the metric to be registered.
            The order is `partials, result_dep1, result_dep2, ...`.
        """

        assert fnc is not None, "No function given for metric {}".format(name)

        if deps is None:
            deps = []
        elif deps == "auto":
            deps = _required_arguments(fnc)[1:]  # first argument is the incremental engine

        if name is None:
            name = fnc.__name__

        if fnc_m is not None:
            if deps_m is None:
                deps_m = []
            elif deps_m == "auto":
                deps_m = _required_arguments(fnc_m)[1:]  # first argument contains per-sequence partials
        else:
            deps_m = None

        self.metrics[name] = {
            "name": name,
            "fnc": fnc,
            "fnc_m": fnc_m,
            "deps": deps,
            "deps_m": deps_m,
            "formatter": formatter,
        }

    @property
    def names(self):
        """Returns the name identifiers of all registered metrics."""
        return [v["name"] for v in self.metrics.values()]

    @property
    def formatters(self):
        """Returns the formatters for all metrics that have associated formatters."""
        return {
            k: v["formatter"]
            for k, v in self.metrics.items()
            if v["formatter"] is not None
        }

    def compute(self, accumulator, metrics=None):
        """Compute metrics and their dependencies from compact state.

        Params
        ------
        accumulator : _Accumulator
            Accumulator containing incremental metric state.

        Kwargs
        ------
        metrics : string, list of string or None, optional
            The identifiers of the metrics to be computed. This method will only
            compute the minimal set of necessary metrics to fullfill the request.
            If None is passed all registered metrics are computed.
        """

        if not isinstance(accumulator, _Accumulator):
            raise TypeError("The metric engine requires an accumulator.")
        if metrics is None:
            metrics = motchallenge_metrics
        elif isinstance(metrics, str):
            metrics = [metrics]

        cache = {}
        engine = accumulator.metrics_engine
        for mname in metrics:
            cache[mname] = self._compute(engine, mname, cache, parent="summarize")
        return cache

    def compute_overall(self, partials, metrics=None):
        """Merge per-sequence primitive dictionaries into overall metrics.

        Params
        ------
        partials : list of metric results to combine overall

        Kwargs
        ------
        metrics : string, list of string or None, optional
            The identifiers of the metrics to be computed. This method will only
            compute the minimal set of necessary metrics to fullfill the request.
            If None is passed all registered metrics are computed.
        """
        if metrics is None:
            metrics = motchallenge_metrics
        elif isinstance(metrics, str):
            metrics = [metrics]
        cache = {}

        for mname in metrics:
            cache[mname] = self._compute_overall(
                partials, mname, cache, parent="summarize"
            )
        return cache

    def _compute(self, engine, name, cache, parent=None):
        """Compute metric and resolve dependencies."""
        assert name in self.metrics, "Cannot find metric {} required by {}.".format(
            name, parent
        )
        already = cache.get(name, None)
        if already is not None:
            return already
        minfo = self.metrics[name]
        vals = []
        for depname in minfo["deps"]:
            v = cache.get(depname, None)
            if v is None:
                v = cache[depname] = self._compute(
                    engine, depname, cache, parent=name
                )
            vals.append(v)
        return minfo["fnc"](engine, *vals)

    def _compute_overall(self, partials, name, cache, parent=None):
        assert name in self.metrics, "Cannot find metric {} required by {}.".format(
            name, parent
        )
        already = cache.get(name, None)
        if already is not None:
            return already
        minfo = self.metrics[name]
        vals = []
        for depname in minfo["deps_m"]:
            v = cache.get(depname, None)
            if v is None:
                v = cache[depname] = self._compute_overall(
                    partials, depname, cache, parent=name
                )
            vals.append(v)
        assert minfo["fnc_m"] is not None, "merge function for metric %s is None" % name
        return minfo["fnc_m"](partials, *vals)


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
    return math_util.quiet_divide(engine.distance_sum, num_detections)


def _merge_motp(partials, num_detections):
    res = 0
    for v in partials:
        res += v["motp"] * v["num_detections"]
    return math_util.quiet_divide(res, num_detections)


def mota(engine, num_misses, num_switches, num_false_positives, num_objects):
    """Multiple object tracker accuracy."""
    del engine  # unused
    return 1.0 - math_util.quiet_divide(
        num_misses + num_switches + num_false_positives, num_objects
    )


def _merge_mota(partials, num_misses, num_switches, num_false_positives, num_objects):
    del partials  # unused
    return 1.0 - math_util.quiet_divide(
        num_misses + num_switches + num_false_positives, num_objects
    )


def precision(engine, num_detections, num_false_positives):
    """Number of detected objects over sum of detected and false positives."""
    del engine  # unused
    return math_util.quiet_divide(num_detections, num_false_positives + num_detections)


def _merge_precision(partials, num_detections, num_false_positives):
    del partials  # unused
    return math_util.quiet_divide(num_detections, num_false_positives + num_detections)


def recall(engine, num_detections, num_objects):
    """Number of detections over number of objects."""
    del engine  # unused
    return math_util.quiet_divide(num_detections, num_objects)


def _merge_recall(partials, num_detections, num_objects):
    del partials  # unused
    return math_util.quiet_divide(num_detections, num_objects)


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
    return math_util.quiet_divide(idtp, idtp + idfp)


def _merge_idp(partials, idtp, idfp):
    del partials  # unused
    return math_util.quiet_divide(idtp, idtp + idfp)


def idr(engine, idtp, idfn):
    """ID measures: global min-cost recall."""
    del engine  # unused
    return math_util.quiet_divide(idtp, idtp + idfn)


def _merge_idr(partials, idtp, idfn):
    del partials  # unused
    return math_util.quiet_divide(idtp, idtp + idfn)


def idf1(engine, idtp, num_objects, num_predictions):
    """ID measures: global min-cost F1 score."""
    del engine  # unused
    return math_util.quiet_divide(2 * idtp, num_objects + num_predictions)


def _merge_idf1(partials, idtp, num_objects, num_predictions):
    del partials  # unused
    return math_util.quiet_divide(2 * idtp, num_objects + num_predictions)


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
    idfp,
    idfn,
    idtp,
}
_MERGE_FUNCTIONS = {
    motp: _merge_motp,
    mota: _merge_mota,
    precision: _merge_precision,
    recall: _merge_recall,
    idp: _merge_idp,
    idr: _merge_idr,
    idf1: _merge_idf1,
}


def _sum_partial(metric):
    metric_name = metric.__name__

    def merge(partials):
        return sum(partial[metric_name] for partial in partials)

    return merge


def _build_metric_host():
    """Build the internal metric dependency engine once per process."""
    m = _MetricsHost()

    def _register(metric, formatter=None):
        merge = _sum_partial(metric) if metric in _ADDITIVE_METRICS else _MERGE_FUNCTIONS.get(metric)
        m._register(metric, formatter=formatter, fnc_m=merge)

    _register(num_frames, formatter="{:d}".format)
    _register(obj_frequencies, formatter="{:d}".format)
    _register(pred_frequencies, formatter="{:d}".format)
    _register(num_matches, formatter="{:d}".format)
    _register(num_switches, formatter="{:d}".format)
    _register(num_transfer, formatter="{:d}".format)
    _register(num_ascend, formatter="{:d}".format)
    _register(num_migrate, formatter="{:d}".format)
    _register(num_false_positives, formatter="{:d}".format)
    _register(num_misses, formatter="{:d}".format)
    _register(num_detections, formatter="{:d}".format)
    _register(num_objects, formatter="{:d}".format)
    _register(num_predictions, formatter="{:d}".format)
    _register(num_gt_ids, formatter="{:d}".format)
    _register(num_dt_ids, formatter="{:d}".format)
    _register(num_unique_objects, formatter="{:d}".format)
    _register(track_ratios)
    _register(mostly_tracked, formatter="{:d}".format)
    _register(partially_tracked, formatter="{:d}".format)
    _register(mostly_lost, formatter="{:d}".format)
    _register(num_fragmentations)
    _register(motp, formatter="{:.3f}".format)
    _register(mota, formatter="{:.1%}".format)
    _register(precision, formatter="{:.1%}".format)
    _register(recall, formatter="{:.1%}".format)

    _register(id_global_assignment)
    _register(idfp)
    _register(idfn)
    _register(idtp)
    _register(idp, formatter="{:.1%}".format)
    _register(idr, formatter="{:.1%}".format)
    _register(idf1, formatter="{:.1%}".format)

    return m


motchallenge_metrics = [
    "idf1",
    "idp",
    "idr",
    "recall",
    "precision",
    "num_unique_objects",
    "mostly_tracked",
    "partially_tracked",
    "mostly_lost",
    "num_false_positives",
    "num_misses",
    "num_switches",
    "num_fragmentations",
    "mota",
    "motp",
    "num_transfer",
    "num_ascend",
    "num_migrate",
]
"""A list of all metrics from MOTChallenge."""

_METRIC_HOST = _build_metric_host()
