"""CLEAR MOT - metrics for multiple object tracker evaluation.

This library provides CLEAR MOT metrics for multiple object tracker evaluation
in Python. The evaluation process is distance metric agnostic making it applicable
to various scenearios (centroid distance, intersection over union and more).

Christoph Heindl, 2017
https://github.com/cheind/py-clearmot
"""

import pandas as pd
import numpy as np
from clearmot.mot import MOTAccumulator
from collections import OrderedDict

def compute_metrics(data):
    if isinstance(data, MOTAccumulator):
        data = data.events

    savediv = lambda a,b: a / b if b != 0 else np.nan

    nframes = float(data.index.levels[0].shape[0])
    nmatch = float(data['Type'].isin(['MATCH']).sum())
    nswitch = float(data['Type'].isin(['SWITCH']).sum())
    nfp = float(data['Type'].isin(['FP']).sum())
    nmiss = float(data['Type'].isin(['MISS']).sum())
    nc = float(nmatch + nswitch)
    ng = float(data['OId'].count())

    objs = data['OId'].value_counts()
    tracked = data[data['Type'] !='MISS']['OId'].value_counts()   
    track_ratio = tracked.div(objs).fillna(1.)

    metr = OrderedDict()
    metr['Frames'] = nframes
    metr['MATCH'] = nmatch
    metr['SWITCH'] = nswitch
    metr['FP'] = nfp
    metr['MISS'] = nmiss
    metr['GT'] = len(data['OId'].value_counts())
    metr['MOTP'] = savediv(data['D'].sum(), nc)
    metr['MOTA'] = 1. - savediv(nmiss + nswitch + nfp, ng)
    metr['PREC'] = savediv(nc, nfp + nc)
    metr['RECALL'] = savediv(nc, ng)
    metr['FAR'] = savediv(nfp, nframes)
    metr['ML'] = track_ratio[track_ratio < 0.2].count()
    metr['PT'] = track_ratio[(track_ratio >= 0.2) & (track_ratio < 0.8)].count()
    metr['MT'] = track_ratio[track_ratio >= 0.8].count()

    return metr

"""
recall=sum(c)/sum(g)*100;
precision=sum(c)/(sum(fp)+sum(c))*100;

percentage of detected targets

% [1]   recall	- recall = percentage of detected targets
% [2]   precision	- precision = percentage of correctly detected targets
% [3]   FAR		- number of false alarms per frame
% [4]   GT        - number of ground truth trajectories
% [5-7] MT, PT, ML	- number of mostly tracked, partially tracked and mostly lost trajectories
% [8]   falsepositives- number of false positives (FP)
% [9]   missed        - number of missed targets (FN)
% [10]  idswitches	- number of id switches     (IDs)
% [11]  FRA       - number of fragmentations
% [12]  MOTA	- Multi-object tracking accuracy in [0,100]
% [13]  MOTP	- Multi-object tracking precision in [0,100] (3D) / [td,100] (2D)
% [14]  MOTAL	- Multi-object tracking accuracy in [0,100] with log10(idswitches)

"""



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

def print_summary(data, names=None, buf=None):
    """Print event statistics for one or more accumulators."""

    df = summarize(data, names=names)
    output = df.to_string(
        buf=buf,
        formatters={
            'MOTA': '{:,.3f}'.format,
            'MOTP': '{:,.3f}'.format,
        }
    )
    print(output)


