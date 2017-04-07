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

def MOTP(data):
    """Returns the multiple object tracking precision."""
    assert isinstance(data, (MOTAccumulator, pd.DataFrame))

    if isinstance(data, MOTAccumulator):
        data = data.events

    denom = data['Type'].isin(('MATCH', 'SWITCH')).sum()
    return data['D'].sum() / denom if denom > 0 else np.nan

def MOTA(data):
    """Returns the multiple object tracking accuracy."""
    assert isinstance(data, (MOTAccumulator, pd.DataFrame))

    if isinstance(data, MOTAccumulator):
        data = data.events

    denom = data['OId'].count()
    return 1. - data['Type'].isin(('MISS', 'SWITCH', 'FP')).sum() / denom if denom > 0 else np.nan


