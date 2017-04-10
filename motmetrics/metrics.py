"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

import pandas as pd
import numpy as np
from collections import OrderedDict, Iterable
from motmetrics.mot import MOTAccumulator

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

