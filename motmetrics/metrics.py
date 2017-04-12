"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

import pandas as pd
import numpy as np
from collections import OrderedDict, Iterable
from motmetrics.mot import MOTAccumulator
import inspect

class MetricsHost:
    def __init__(self):
        self.metrics = {}

    def register(self, fnc, deps='auto', name=None, helpstr=None, formatter=None):
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
        return [v['name'] for v in self.metrics.values()]

    def list_metrics(self, include_deps=False):
        cols = ['Name', 'Description', 'Dependencies']
        if include_deps:
            data = [(m['name'], m['help'], m['deps']) for m in self.metrics.values()]
        else:
            data = [(m['name'], m['help']) for m in self.metrics.values()]
            cols = cols[:-1]

        return pd.DataFrame(data, columns=cols)

    def to_markdown(self, include_deps=False):
        df = self.list_metrics(include_deps=include_deps)
        fmt = [':---' for i in range(len(df.columns))]
        df_fmt = pd.DataFrame([fmt], columns=df.columns)
        df_formatted = pd.concat([df_fmt, df])
        return df_formatted.to_csv(sep="|", index=False)

    def compute(self, df, metrics=None, return_dataframe=True, name=None):        
        
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

    def _compute(self, df, name, cache, parent=None):
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
    return float(df.index.get_level_values(0).unique().shape[0])

def obj_frequencies(df):
    """Total number of occurrences of individual objects."""
    return df.OId.value_counts()

def num_unique_objects(df, obj_frequencies):
    """Total number of unique object ids encountered."""
    return float(len(obj_frequencies))

def num_matches(df):
    """Total number matches."""
    return float(df.Type.isin(['MATCH']).sum())

def num_switches(df):
    """Total number of track switches."""
    return float(df.Type.isin(['SWITCH']).sum())

def num_false_positives(df):
    """Total number of false positives (false-alarms)."""
    return float(df.Type.isin(['FP']).sum())

def num_misses(df):
    """Total number of misses."""
    return float(df.Type.isin(['MISS']).sum())

def num_detections(df, num_matches, num_switches):
    """Total number of detected objects including matches and switches."""
    return num_matches + num_switches

def num_objects(df):
    """Total number of objects."""
    return float(df.OId.count())

def track_ratios(df, obj_frequencies):
    """Ratio of assigned to total appearance count per unique object id."""   
    tracked = df[df.Type !='MISS']['OId'].value_counts()   
    return tracked.div(obj_frequencies).fillna(1.)

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

def default_metrics():
    m = MetricsHost()

    m.register(num_frames, formatter='{:d}'.format)
    m.register(obj_frequencies, formatter='{:d}'.format)    
    m.register(num_matches, formatter='{:d}'.format)
    m.register(num_switches, formatter='{:d}'.format)
    m.register(num_false_positives, formatter='{:d}'.format)
    m.register(num_misses, formatter='{:d}'.format)
    m.register(num_detections, formatter='{:d}'.format)
    m.register(num_objects, formatter='{:d}'.format)
    m.register(num_unique_objects, deps='auto', formatter='{:d}'.format)
    m.register(track_ratios, deps='auto')
    m.register(mostly_tracked, deps='auto', formatter='{:d}'.format)
    m.register(partially_tracked, deps='auto', formatter='{:d}'.format)
    m.register(mostly_lost, deps='auto', formatter='{:d}'.format)
    m.register(num_fragmentations, deps='auto')
    m.register(motp, deps='auto', formatter='{:.3f}'.format)
    m.register(mota, deps='auto', formatter='{:.2%}'.format)
    m.register(precision, deps='auto', formatter='{:.2%}'.format)
    m.register(recall, deps='auto', formatter='{:.2%}'.format)

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
