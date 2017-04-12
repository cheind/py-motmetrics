"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

import pandas as pd
import numpy as np
from collections import OrderedDict, Iterable
from motmetrics.mot import MOTAccumulator
import inspect

class MetricsContainer:
    def __init__(self):
        self.metrics = {}

    def register(self, fnc, deps=None, name=None, helpstr=None, formatter=None):
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

    def compute(self, df, metrics=None, name=None):
        cache = {}

        if metrics is None:
            metrics = self.names

        for mname in metrics:
            cache[mname] = self._compute(df, mname, cache, parent='summarize')            

        if name is None:
            name = 0 

        data = OrderedDict([(k, cache[k]) for k in metrics])
        return pd.DataFrame(data, index=[name])

    def _compute(self, df, name, cache, parent=None):
        assert name in self.metrics, 'Cannot find metric {} required by {}.'.format(name, parent)
        minfo = self.metrics[name]
        vals = []
        for depname in minfo['deps']:
            if not depname in cache:
                cache[depname] = self._compute(df, depname, cache, parent=name)
            vals.append(cache[depname])
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
    """Total number matches.
    
    Multiline test.
    """
    return float(df.Type.isin(['MATCH']).sum())

def num_switches(df):
    """Total number of track switches."""
    return float(df.Type.isin(['SWITCH']).sum())

def num_falsepositives(df):
    """Total number of false positives (false-alarms)."""
    return float(df.Type.isin(['FP']).sum())

def num_misses(df):
    """Total number of misses."""
    float(df.Type.isin(['MISS']).sum())

def num_detections(df, num_matches, num_switches):
    """Total number of detected objects including matches and switches."""
    return num_matches + num_switches

def num_objects(df):
    """Total number of objects."""
    return float(df.OId.count())

def track_ratios(df, obj_frequencies):
    """Ratio of assigned to total appearance count per unique object id."""   
    tracked = data[df.Type !='MISS']['OId'].value_counts()   
    return tracked.div(obj_frequencies).fillna(1.)

def mostly_tracked(df, track_ratios):
    """Number of objects tracked for at least 80 percent of lifespan."""
    return track_ratio[track_ratios >= 0.8].count()

def partially_tracked(df, track_ratios):
    """Number of objects tracked between 20 and 80 percent of lifespan."""
    return track_ratios[(track_ratios >= 0.2) & (track_ratios < 0.8)].count()

def mostly_lost(df, track_ratios):
    """Number of objects tracked less than 20 percent of lifespan."""
    return track_ratio[track_ratio < 0.2].count()

def num_fragmentation(df, obj_frequencies):
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

def mota(df, num_misses, num_switches, num_falsepositives, num_objects):
    """Multiple object tracker accuracy."""
    return 1. - (num_misses + num_switches + num_falsepositives) / num_objects

def precision(df, num_detections, num_falsepositives):
    """Number of detected objects over sum of detected and false positives."""
    return num_detections / (num_falsepositives + num_detections)

def recall(df, num_detections, num_objects):
    """Number of detections over number of objects."""
    return num_detections / num_objects

def default_metrics():
    m = MetricsContainer()

    m.register(num_frames, formatter='{:d}'.format)
    m.register(obj_frequencies, formatter='{:d}'.format)    
    m.register(num_matches, formatter='{:d}'.format)
    m.register(num_switches, formatter='{:d}'.format)
    m.register(num_falsepositives, formatter='{:d}'.format)
    m.register(num_misses, formatter='{:d}'.format)
    m.register(num_detections, formatter='{:d}'.format)
    m.register(num_objects, formatter='{:d}'.format)
    m.register(num_unique_objects, deps='auto', formatter='{:d}'.format)
    m.register(track_ratios, deps='auto')
    m.register(mostly_tracked, deps='auto', formatter='{:d}'.format)
    m.register(partially_tracked, deps='auto', formatter='{:d}'.format)
    m.register(mostly_lost, deps='auto', formatter='{:d}'.format)
    m.register(num_fragmentation, deps='auto')
    m.register(motp, deps='auto', formatter='{:.3f}'.format)
    m.register(mota, deps='auto', formatter='{:.2%}'.format)
    m.register(precision, deps='auto', formatter='{:.2%}'.format)
    m.register(recall, deps='auto', formatter='{:.2%}'.format)

    return m



def compute_metrics(data):
    """Returns computed metrics for event data frame.

    Params
    ------
    data : pd.DataFrame or MOTAccumulator
        Events data frame to compute metrics for.
    
    Returns
    -------
    metr : dict
        Dictionary of computed metrics. Currently the following metrics are computed as fields
        in the dictionary

        - `Frames` total number of frames
        - `Match` total number of matches
        - `Switch` total number of track switches
        - `FalsePos` total number of false positives, i.e false alarms
        - `Miss` total number of misses
        - `MOTA` Tracker accuracy as defined in [1]
        - `MOTP` Tracker precision as defined in [1]. Since motmetrics is distance agnostic,
        this value depends on the distance and threshold on distance used. To compare this value to
        results from MOTChallenge 2D use 1.-MOTP
        - `Precision` Percent of correct detections to total tracker detections
        - `Recall` Percent of correct detections to total number of objects
        - `Frag` Number of track fragmentations as defined in [2]
        - `Objs` Total number of unique objects
        - `MT` Number of mostly tracked targets as defined in [2,3]
        - `PT` Number of partially tracked targets as defined in [2,3]
        - `ML´ Number of mostly lost targets as defined in [2, 3]

    References
    ----------
    1. Bernardin, Keni, and Rainer Stiefelhagen. "Evaluating multiple object tracking performance: the CLEAR MOT metrics." 
    EURASIP Journal on Image and Video Processing 2008.1 (2008): 1-10.
    2. Milan, Anton, et al. "Mot16: A benchmark for multi-object tracking." arXiv preprint arXiv:1603.00831 (2016).
    3. Li, Yuan, Chang Huang, and Ram Nevatia. "Learning to associate: Hybridboosted multi-target tracker for crowded scene." 
    Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on. IEEE, 2009.
    """

    if isinstance(data, MOTAccumulator):
        data = data.events

    savediv = lambda a,b: a / b if b != 0 else np.nan

    # Common values
    
    nframes = float(data.index.get_level_values(0).unique().shape[0]) # Works for Dataframes and slices
    nmatch = float(data.Type.isin(['MATCH']).sum())
    nswitch = float(data.Type.isin(['SWITCH']).sum())
    nfp = float(data.Type.isin(['FP']).sum())
    nmiss = float(data.Type.isin(['MISS']).sum())
    nc = float(nmatch + nswitch)
    ng = float(data['OId'].count())

    # Compute MT, PT, ML
    # First count for each object the number of total occurrences. Next count for each object the 
    # number of times a correspondence was assigned. The track ratio corresponds to assigned / total 
    # for each object separately. Finally classify into MT, PT, ML (see further below).
    # Also account for cases when an object is never missed (fillna below).
    objs = data['OId'].value_counts()
    tracked = data[data.Type !='MISS']['OId'].value_counts()   
    track_ratio = tracked.div(objs).fillna(1.)

    # Compute fragmentation
    fra = 0
    for o in objs.index:
        # Find first and last time object was not missed (track span). Then count
        # the number switches from NOT MISS to MISS state.
        dfo = data[data.OId == o]
        notmiss = dfo[dfo.Type != 'MISS']
        if len(notmiss) == 0:
            continue
        first = notmiss.index[0]
        last = notmiss.index[-1]
        diffs = dfo.loc[first:last].Type.apply(lambda x: 1 if x == 'MISS' else 0).diff()
        fra += diffs[diffs == 1].count()
        
    metr = OrderedDict() # Use ordered dict to column order is preserved.
    metr['Frames'] = int(nframes)
    metr['Match'] = int(nmatch)
    metr['Switch'] = int(nswitch)
    metr['FalsePos'] = int(nfp)
    metr['Miss'] = int(nmiss)
    metr['MOTP'] = savediv(data['D'].sum(), nc)
    metr['MOTA'] = 1. - savediv(nmiss + nswitch + nfp, ng)
    metr['Precision'] = savediv(nc, nfp + nc)
    metr['Recall'] = savediv(nc, ng)
    metr['Frag'] = fra
    metr['Objs'] = len(objs)        
    metr['MT'] = track_ratio[track_ratio >= 0.8].count()
    metr['PT'] = track_ratio[(track_ratio >= 0.2) & (track_ratio < 0.8)].count()
    metr['ML'] = track_ratio[track_ratio < 0.2].count()

    return metr

def summarize(accs, names=None):
    """Compute event statistics of one or more MOT accumulators.
    
    Params
    ------
    accs : MOTAccumulator or list thereof
        Event accumulators to summarize.

    Kwargs
    ------
    names : string or list thereof, optional
        Name for accumulators

    Returns
    -------
    summary : pd.DataFrame
        A dataframe having metrics in columns and accumulator
        results in rows (one per accumulator). See `compute_metrics`
        for docs on available metrics.
    """
    
    if isinstance(accs, (MOTAccumulator, pd.DataFrame)):
        accs = [accs]

    if names is None:
        names = list(range(len(accs)))
    elif not isinstance(names, Iterable):
        names = [names]

    events = []
    for idx, d in enumerate(accs):
        events.append(d.events if isinstance(d, MOTAccumulator) else d)  
        
    dfs = []
    for name, ev in zip(names, events):
        dfs.append(pd.DataFrame(compute_metrics(ev), index=[name]))
    return pd.concat(dfs)

