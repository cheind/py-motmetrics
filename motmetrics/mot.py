"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

import numpy as np
import numpy.ma as ma
import pandas as pd
from collections import OrderedDict, namedtuple
from itertools import count
from scipy.optimize import linear_sum_assignment


MOTAccumulator = namedtuple('MOTAccumulator', ['events', 'm', 'auto_id'], verbose=False)
"""Defines the data type for storing MOT related events"""

def new_dataframe():
    idx = pd.MultiIndex(levels=[[],[]], labels=[[],[]], names=['FrameId','Event'])
    cats = pd.Categorical([], categories=['FP', 'MISS', 'SWITCH', 'MATCH'])
    df = pd.DataFrame(
        OrderedDict([
            ('Type', pd.Series(cats)),          # Type of event. One of FP (false positive), MISS, SWITCH, MATCH
            ('OId', pd.Series(dtype=str)),      # Object ID or -1 if FP. Using float as missing values will be converted to NaN anyways.
            ('HId', pd.Series(dtype=str)),      # Hypothesis ID or NaN if MISS. Using float as missing values will be converted to NaN anyways.
            ('D', pd.Series(dtype=float)),      # Distance or NaN when FP or MISS            
        ]),
        index=idx
    )
    return df

def new_accumulator(auto_id=False):
    """Returns a new accumulator to store MOT related data."""
    return MOTAccumulator(events=new_dataframe(), m={}, auto_id=auto_id)      

def update(acc, oids, hids, dists, frameid=None):
    """Update the accumulator with frame specific objects/detections."""
        
    oids = ma.array(oids, mask=np.zeros(len(oids)))
    hids = ma.array(hids, mask=np.zeros(len(hids)))  
    dists = np.atleast_2d(dists).astype(float).reshape(oids.shape[0], hids.shape[0])

    if frameid is None:
        assert acc.auto_id, 'Auto-increment is not enabled'
        frameid = acc.events.index.get_level_values(0).shape[0]
    
    eid = count()
    dists, INVDIST = _sanitize_dists(dists)

    if oids.size * hids.size > 0:        
        # 1. Try to re-establish tracks from previous correspondences
        for i in range(oids.shape[0]):
            if not oids[i] in acc.m:
                continue

            hprev = acc.m[oids[i]]                    
            j, = np.where(hids==hprev)  
            if j.shape[0] == 0:
                continue
            j = j[0]

            if not dists[i, j] == INVDIST:
                oids[i] = ma.masked
                hids[j] = ma.masked
                acc.m[oids.data[i]] = hids.data[j]
                acc.events.loc[(frameid, next(eid)), :] = ['MATCH', oids.data[i], hids.data[j], dists[i, j]]
        
        # 2. Try to remaining objects/hypotheses
        dists[oids.mask, :] = INVDIST
        dists[:, hids.mask] = INVDIST
    
        rids, cids = linear_sum_assignment(dists)
        for i, j in zip(rids, cids):

            if oids[i] is ma.masked or hids[j] is ma.masked or dists[i, j] == INVDIST:
                continue
            
            cat = 'SWITCH' if oids[i] in acc.m and not acc.m[oids[i]] == hids.data[j] else 'MATCH'
            acc.events.loc[(frameid, next(eid)), :] = [cat, oids.data[i], hids.data[j], dists[i, j]]
            oids[i] = ma.masked
            hids[j] = ma.masked
            acc.m[oids.data[i]] = hids.data[j]

    # 3. All remaining objects are missed
    for o in oids[~oids.mask]:
        acc.events.loc[(frameid, next(eid)), :] = ['MISS', o, np.nan, np.nan]
    
    # 4. All remaining hypotheses are false alarms
    for h in hids[~hids.mask]:
        acc.events.loc[(frameid, next(eid)), :] = ['FP', np.nan, h, np.nan]

def _sanitize_dists(dists):
    dists = np.copy(dists)
    
    # Note there is an issue in scipy.optimize.linear_sum_assignment where
    # it runs forever if an entire row/column is infinite or nan. We therefore
    # make a copy of the distance matrix and compute a safe value that indicates
    # 'cannot assign'. Also note + 1 is necessary in below inv-dist computation
    # to make invdist bigger than max dist in case max dist is zero.
    
    valid_dists = dists[np.isfinite(dists)]
    INVDIST = 2 * valid_dists.max() + 1 if valid_dists.shape[0] > 0 else 1.
    dists[~np.isfinite(dists)] = INVDIST  

    return dists, INVDIST


        
        
