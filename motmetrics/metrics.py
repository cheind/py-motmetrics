"""CLEAR MOT - metrics for multiple object tracker evaluation.

This library provides CLEAR MOT metrics for multiple object tracker evaluation
in Python. The evaluation process is distance metric agnostic making it applicable
to various scenearios (centroid distance, intersection over union and more).

Christoph Heindl, 2017
https://github.com/cheind/py-clearmot
"""

import pandas as pd
import numpy as np
from collections import OrderedDict, Iterable
from motmetrics.mot import MOTAccumulator

def compute_metrics(data):
    if isinstance(data, MOTAccumulator):
        data = data.events

    savediv = lambda a,b: a / b if b != 0 else np.nan

    nframes = float(data.index.get_level_values(0).shape[0])
    nmatch = float(data['Type'].isin(['MATCH']).sum())
    nswitch = float(data['Type'].isin(['SWITCH']).sum())
    nfp = float(data['Type'].isin(['FP']).sum())
    nmiss = float(data['Type'].isin(['MISS']).sum())
    nc = float(nmatch + nswitch)
    ng = float(data['OId'].count())

    # Compute MT, PT, ML
    objs = data['OId'].value_counts()
    tracked = data[data['Type'] !='MISS']['OId'].value_counts()   
    track_ratio = tracked.div(objs).fillna(1.)

    # Compute fragmentation
    fra = 0
    for o in objs.index:
        dfo = data[data.OId == o]
        notmiss = dfo[dfo.Type != 'MISS']
        if len(notmiss) == 0:
            continue
        first = notmiss.index[0]
        last = notmiss.index[-1]
        diffs = dfo.loc[first:last].Type.apply(lambda x: 1 if x == 'MISS' else 0).diff()
        fra += diffs[diffs == 1].count()
        
    metr = OrderedDict()
    metr['Frames'] = int(nframes)
    metr['Matches'] = int(nmatch)
    metr['Switches'] = int(nswitch)
    metr['FalsePos'] = int(nfp)
    metr['Misses'] = int(nmiss)
    metr['MOTP'] = savediv(data['D'].sum(), nc)
    metr['MOTA'] = 1. - savediv(nmiss + nswitch + nfp, ng)
    metr['Precision'] = savediv(nc, nfp + nc)
    metr['Recall'] = savediv(nc, ng)
    metr['Frag'] = fra
    metr['Objs'] = len(data['OId'].value_counts())        
    metr['MT'] = track_ratio[track_ratio >= 0.8].count()
    metr['PT'] = track_ratio[(track_ratio >= 0.2) & (track_ratio < 0.8)].count()
    metr['ML'] = track_ratio[track_ratio < 0.2].count()

    return metr

def summarize(accs, names=None):
    """Compute event statistics of one or more MOT accumulators."""
    
    if isinstance(accs, MOTAccumulator):
        accs = [accs]

    if names is None:
        names = list(range(len(accs)))
    elif not isinstance(names, Iterable):
        names = [names]

    events = []
    for idx, d in enumerate(accs):
        assert isinstance(d, (MOTAccumulator, pd.DataFrame)) 
        events.append(d.events if isinstance(d, MOTAccumulator) else d)  
        
    dfs = []
    for name, ev in zip(names, events):
        dfs.append(pd.DataFrame(compute_metrics(ev), index=[name]))
    return pd.concat(dfs)

def print_summary(summary):
    """Print event statistics for one or more accumulators."""
    output = summary.to_string(
        buf=buf,
        formatters={
            'MOTA': '{:,.3f}'.format,
            'MOTP': '{:,.3f}'.format,
            'PREC': '{:,.3f}'.format,
            'RECALL': '{:,.3f}'.format,
        }
    )
    print(output)

