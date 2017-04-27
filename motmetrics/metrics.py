"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

from __future__ import division
from collections import OrderedDict, Iterable
from motmetrics.mot import MOTAccumulator
import pandas as pd
import numpy as np
import inspect

class MetricsHost:
    """Keeps track of metrics and intra metric dependencies."""

    def __init__(self):
        self.metrics = OrderedDict()

    def register(self, fnc, deps='auto', name=None, helpstr=None, formatter=None):
        """Register a new metric.

        Params
        ------
        fnc : Function
            Function that computes the metric to be registered. The number of arguments
            is 1 + N, where N is the number of dependencies of the metric to be registered.
            The order of the argument passed is `df, result_dep1, result_dep2, ...`.

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
        helpstr : string or None, optional 
            A description of what the metric computes. If no help message is given it
            is deduced from the docstring of the function.
        formatter: Format object, optional
            An optional default formatter when rendering metric results as string. I.e to
            render the result `0.35` as `35%` one would pass `{:.2%}.format`
        """        

        assert not fnc is None, 'No function given for metric {}'.format(name)

        if deps is None:
            deps = []
        elif deps is 'auto':            
            deps = inspect.getargspec(fnc).args[1:] # assumes dataframe as first argument

        if name is None:
            name = fnc.__name__ # Relies on meaningful function names, i.e don't use for lambdas

        if helpstr is None:
            helpstr = inspect.getdoc(fnc) if inspect.getdoc(fnc) else 'No description.'
            helpstr = ' '.join(helpstr.split())
            
        self.metrics[name] = {
            'name' : name,
            'fnc' : fnc,
            'deps' : deps,
            'help' : helpstr,
            'formatter' : formatter
        }

    @property
    def names(self):
        """Returns the name identifiers of all registered metrics."""
        return [v['name'] for v in self.metrics.values()]
    
    @property
    def formatters(self):
        """Returns the formatters for all metrics that have associated formatters."""
        return dict([(k, v['formatter']) for k, v in self.metrics.items() if not v['formatter'] is None])

    def list_metrics(self, include_deps=False):
        """Returns a dataframe containing names, descriptions and optionally dependencies for each metric."""
        cols = ['Name', 'Description', 'Dependencies']
        if include_deps:
            data = [(m['name'], m['help'], m['deps']) for m in self.metrics.values()]
        else:
            data = [(m['name'], m['help']) for m in self.metrics.values()]
            cols = cols[:-1]

        return pd.DataFrame(data, columns=cols)

    def list_metrics_markdown(self, include_deps=False):
        """Returns a markdown ready version of `list_metrics`."""
        df = self.list_metrics(include_deps=include_deps)
        fmt = [':---' for i in range(len(df.columns))]
        df_fmt = pd.DataFrame([fmt], columns=df.columns)
        df_formatted = pd.concat([df_fmt, df])
        return df_formatted.to_csv(sep="|", index=False)

    def compute(self, df, metrics=None, return_dataframe=True, name=None):
        """Compute metrics on the dataframe / accumulator.
        
        Params
        ------
        df : MOTAccumulator or pandas.DataFrame
            The dataframe to compute the metrics on
        
        Kwargs
        ------
        metrics : string, list of string or None, optional
            The identifiers of the metrics to be computed. This method will only
            compute the minimal set of necessary metrics to fullfill the request.
            If None is passed all registered metrics are computed.
        return_dataframe : bool, optional
            Return the result as pandas.DataFrame (default) or dict.
        name : string, optional
            When returning a pandas.DataFrame this is the index of the row containing
            the computed metric values.
        """ 
        
        if isinstance(df, MOTAccumulator):
            df = df.events

        if metrics is None:
            metrics = self.names
        elif isinstance(metrics, str):
            metrics = [metrics]

        cache = {}
        for mname in metrics:
            cache[mname] = self._compute(df, mname, cache, parent='summarize')            

        if name is None:
            name = 0 

        data = OrderedDict([(k, cache[k]) for k in metrics])
        return pd.DataFrame(data, index=[name]) if return_dataframe else data     

    def compute_many(self, dfs, metrics=None, names=None):
        """Compute metrics on multiple dataframe / accumulators.
        
        Params
        ------
        dfs : list of MOTAccumulator or list of pandas.DataFrame
            The data to compute metrics on.
        
        Kwargs
        ------
        metrics : string, list of string or None, optional
            The identifiers of the metrics to be computed. This method will only
            compute the minimal set of necessary metrics to fullfill the request.
            If None is passed all registered metrics are computed.
        names : list of string, optional
            The names of individual rows in the resulting dataframe.

        Returns
        -------
        df : pandas.DataFrame
            A datafrom containing the metrics in columns and names in rows.
        """

        assert names is None or len(names) == len(dfs)

        if names is None:
            names = range(len(dfs))

        partials = [self.compute(acc, metrics=metrics, name=name) for acc, name in zip(dfs, names)]
        return pd.concat(partials)


    def _compute(self, df, name, cache, parent=None):
        """Compute metric and resolve dependencies."""
        assert name in self.metrics, 'Cannot find metric {} required by {}.'.format(name, parent)
        minfo = self.metrics[name]
        vals = []
        for depname in minfo['deps']:
            v = cache.get(depname, None)
            if v is None:
                v = cache[depname] = self._compute(df, depname, cache, parent=name)
            vals.append(v)
        return minfo['fnc'](df, *vals)

def num_frames(df):
    """Total number of frames."""
    return df.index.get_level_values(0).unique().shape[0]

def obj_frequencies(df):
    """Total number of occurrences of individual objects."""
    return df.OId.value_counts()

def num_unique_objects(df, obj_frequencies):
    """Total number of unique object ids encountered."""
    return len(obj_frequencies)

def num_matches(df):
    """Total number matches."""
    return df.Type.isin(['MATCH']).sum()

def num_switches(df):
    """Total number of track switches."""
    return df.Type.isin(['SWITCH']).sum()

def num_false_positives(df):
    """Total number of false positives (false-alarms)."""
    return df.Type.isin(['FP']).sum()

def num_misses(df):
    """Total number of misses."""
    return df.Type.isin(['MISS']).sum()

def num_detections(df, num_matches, num_switches):
    """Total number of detected objects including matches and switches."""
    return num_matches + num_switches

def num_objects(df):
    """Total number of objects."""
    return df.OId.count()

def track_ratios(df, obj_frequencies):
    """Ratio of assigned to total appearance count per unique object id."""   
    tracked = df[df.Type != 'MISS']['OId'].value_counts()
    return tracked.div(obj_frequencies).fillna(0.)

def mostly_tracked(df, track_ratios):
    """Number of objects tracked for at least 80 percent of lifespan."""
    return track_ratios[track_ratios >= 0.8].count()

def partially_tracked(df, track_ratios):
    """Number of objects tracked between 20 and 80 percent of lifespan."""
    return track_ratios[(track_ratios >= 0.2) & (track_ratios < 0.8)].count()

def mostly_lost(df, track_ratios):
    """Number of objects tracked less than 20 percent of lifespan."""
    return track_ratios[track_ratios < 0.2].count()

def num_fragmentations(df, obj_frequencies):
    """Total number of switches from tracked to not tracked."""
    fra = 0
    for o in obj_frequencies.index:
        # Find first and last time object was not missed (track span). Then count
        # the number switches from NOT MISS to MISS state.
        dfo = df[df.OId == o]
        notmiss = dfo[dfo.Type != 'MISS']
        if len(notmiss) == 0:
            continue
        first = notmiss.index[0]
        last = notmiss.index[-1]
        diffs = dfo.loc[first:last].Type.apply(lambda x: 1 if x == 'MISS' else 0).diff()
        fra += diffs[diffs == 1].count()
    return fra

def motp(df, num_detections):
    """Multiple object tracker precision."""
    return df['D'].sum() / num_detections

def mota(df, num_misses, num_switches, num_false_positives, num_objects):
    """Multiple object tracker accuracy."""
    return 1. - (num_misses + num_switches + num_false_positives) / num_objects

def precision(df, num_detections, num_false_positives):
    """Number of detected objects over sum of detected and false positives."""
    return num_detections / (num_false_positives + num_detections)

def recall(df, num_detections, num_objects):
    """Number of detections over number of objects."""
    return num_detections / num_objects

def create():
    """Creates a MetricsHost and populates it with default metrics."""
    m = MetricsHost()

    m.register(num_frames, formatter='{:d}'.format)
    m.register(obj_frequencies, formatter='{:d}'.format)    
    m.register(num_matches, formatter='{:d}'.format)
    m.register(num_switches, formatter='{:d}'.format)
    m.register(num_false_positives, formatter='{:d}'.format)
    m.register(num_misses, formatter='{:d}'.format)
    m.register(num_detections, formatter='{:d}'.format)
    m.register(num_objects, formatter='{:d}'.format)
    m.register(num_unique_objects, formatter='{:d}'.format)
    m.register(track_ratios)
    m.register(mostly_tracked, formatter='{:d}'.format)
    m.register(partially_tracked, formatter='{:d}'.format)
    m.register(mostly_lost, formatter='{:d}'.format)
    m.register(num_fragmentations)
    m.register(motp, formatter='{:.3f}'.format)
    m.register(mota, formatter='{:.2%}'.format)
    m.register(precision, formatter='{:.2%}'.format)
    m.register(recall, formatter='{:.2%}'.format)

    return m

motchallenge_metrics = [
    'recall', 
    'precision', 
    'num_unique_objects', 
    'mostly_tracked', 
    'partially_tracked', 
    'mostly_lost', 
    'num_false_positives', 
    'num_misses',
    'num_switches',
    'num_fragmentations',
    'mota',
    'motp'
]
"""A list of all metrics from MOTChallenge."""
