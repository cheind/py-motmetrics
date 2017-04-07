"""CLEAR MOT - metrics for multiple object tracker evaluation.

This library provides CLEAR MOT metrics for multiple object tracker evaluation
in Python. The evaluation process is distance metric agnostic making it applicable
to various scenearios (centroid distance, intersection over union and more).

Christoph Heindl, 2017
https://github.com/cheind/py-clearmot
"""

import pandas as pd
from collections import OrderedDict
from collections import Iterable

from clearmot.mot import MOTAccumulator
from clearmot.metrics import MOTA, MOTP

def compute_stats(accs, names=None):
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
        dfs.append(pd.DataFrame(OrderedDict([
            ('Events', ev.shape[0]),
            ('MATCH', ev['Type'].isin(['MATCH']).sum()),
            ('SWITCH', ev['Type'].isin(['SWITCH']).sum()),
            ('FP', ev['Type'].isin(['FP']).sum()),
            ('MISS', ev['Type'].isin(['MISS']).sum()),        
            ('MOTP', MOTP(ev)),
            ('MOTA', MOTA(ev))
        ]), index=[name]))

    return pd.concat(dfs)

def print_stats(data, names=None):
    """Print event statistics for one or more MOT accumulators."""
    print(compute_stats(data, names=names))
