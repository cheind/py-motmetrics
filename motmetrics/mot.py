"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

import numpy as np
import numpy.ma as ma
import pandas as pd
from collections import OrderedDict
from itertools import count
from scipy.optimize import linear_sum_assignment

class MOTAccumulator(object):
    """Manage tracking events.
    
    This class computes per-frame tracking events from a given set of object / hypothesis 
    ids and pairwise distances. Indended usage

        import motmetrics as mm
        acc = mm.MOTAccumulator()
        acc.update(['a', 'b'], [0, 1, 2], dists, frameid=0)
        ...
        acc.update(['d'], [6,10], other_dists, frameid=76)        
        summary = mm.metrics.summarize(acc)
        print(mm.io.render_summary(summary))

    Update is called once per frame and takes objects / hypothesis ids and a pairwise distance
    matrix between those (see distances module for support). Per frame max(len(objects), len(hypothesis)) 
    events are generated. Each event type is one of the following
        - `'MATCH'` a match between a object and hypothesis was found
        - `'SWITCH'` a match between a object and hypothesis was found but differs from previous assignment
        - `'MISS'` no match for an object was found
        - `'FP'` no match for an hypothesis was found (spurious detections)
    
    Events are tracked in a pandas Dataframe. The dataframe is hierarchically indexed by (`FrameId`, `EventId`),
    where `FrameId` is either provided during the call to `update` or auto-incremented when `auto_id` is set
    true during construction of MOTAccumulator. `EventId` is auto-incremented. The dataframe has the following
    columns 
        - `Type` one of `('MATCH', 'SWITCH', 'MISS', 'FP')`
        - `OId` object id or np.nan when `'FP'`
        - `HId` hypothesis id or np.nan when `'MISS'`
        - `D` distance or np.nan when `'FP'` or `'MISS'`
    
    From the events and associated fields the entire tracking history can be recovered. Once the accumulator 
    has been populated with per-frame data use `metrics.summarize` to compute statistics. See `metrics.compute_metrics`
    for a list of metrics computed.

    References
    ----------
    1. Bernardin, Keni, and Rainer Stiefelhagen. "Evaluating multiple object tracking performance: the CLEAR MOT metrics." 
    EURASIP Journal on Image and Video Processing 2008.1 (2008): 1-10.
    2. Milan, Anton, et al. "Mot16: A benchmark for multi-object tracking." arXiv preprint arXiv:1603.00831 (2016).
    3. Li, Yuan, Chang Huang, and Ram Nevatia. "Learning to associate: Hybridboosted multi-target tracker for crowded scene." 
    Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on. IEEE, 2009.
    """

    def __init__(self, auto_id=False, max_switch_time=float('inf')):
        """Create a MOTAccumulator.

        Params
        ------
        auto_id : bool, optional
            Whether or not frame indices are auto-incremented or provided upon
            updating. Defaults to false. Not specifying a frame-id when this value
            is true results in an error. Specifying a frame-id when this value is
            false also results in an error.

        max_switch_time : scalar, optional
            Allows specifying an upper bound on the timespan an unobserved but 
            tracked object is allowed to generate track switch events. Useful if groundtruth 
            objects leaving the field of view keep their ID when they reappear, 
            but your tracker is not capable of recognizing this (resulting in 
            track switch events). The default is that there is no upper bound
            on the timespan. In units of frame timestamps. When using auto_id
            in units of count.
        """

        self.auto_id = auto_id
        self.max_switch_time = max_switch_time
        self.reset()       

    def reset(self):
        """Reset the accumulator to empty state."""

        self.events = MOTAccumulator.new_event_dataframe()
        self.m = {} # Pairings up to current timestamp  
        self.last_occurrence = {} # Tracks most recent occurance of object

    def update(self, oids, hids, dists, frameid=None):
        """Updates the accumulator with frame specific objects/detections.

        This method generates events based on the following algorithm [1]:
        1. Try to carry forward already established tracks. If any paired object / hypothesis
        from previous timestamps are still visible in the current frame, create a 'MATCH' 
        event between them.
        2. For the remaining constellations minimize the total object / hypothesis distance
        error (Kuhn-Munkres algorithm). If a correspondence made contradicts a previous
        match create a 'SWITCH' else a 'MATCH' event.
        3. Create 'MISS' events for all remaining unassigned objects.
        4. Create 'FP' events for all remaining unassigned hypotheses.
        
        Params
        ------
        oids : N array 
            Array of object ids.
        hids : M array 
            Array of hypothesis ids.
        dists: NxM array
            Distance matrix. np.nan values to signal do-not-pair constellations.
            See `distances` module for support methods.  

        Kwargs
        ------
        frameId : id
            Unique frame id. Optional when MOTAccumulator.auto_id is specified during
            construction.

        Returns
        -------
        frame_events : pd.DataFrame
            Dataframe containing generated events

        References
        ----------
        1. Bernardin, Keni, and Rainer Stiefelhagen. "Evaluating multiple object tracking performance: the CLEAR MOT metrics." 
        EURASIP Journal on Image and Video Processing 2008.1 (2008): 1-10.
        """
            
        oids = ma.array(oids, mask=np.zeros(len(oids)))
        hids = ma.array(hids, mask=np.zeros(len(hids)))  
        dists = np.atleast_2d(dists).astype(float).reshape(oids.shape[0], hids.shape[0])

        if frameid is None:            
            assert self.auto_id, 'auto-id is not enabled'
            frameid = self.events.index.get_level_values(0).unique().shape[0]   
        else:
            assert not self.auto_id, 'Cannot provide frame id when auto-id is enabled'
        
        eid = count()
        dists, INVDIST = self._sanitize_dists(dists)

        if oids.size * hids.size > 0:        
            # 1. Try to re-establish tracks from previous correspondences
            for i in range(oids.shape[0]):
                if not oids[i] in self.m:
                    continue

                hprev = self.m[oids[i]]                    
                j, = np.where(hids==hprev)  
                if j.shape[0] == 0:
                    continue
                j = j[0]

                if not dists[i, j] == INVDIST:
                    oids[i] = ma.masked
                    hids[j] = ma.masked
                    self.m[oids.data[i]] = hids.data[j]
                    self.events.loc[(frameid, next(eid)), :] = ['MATCH', oids.data[i], hids.data[j], dists[i, j]]
            
            # 2. Try to remaining objects/hypotheses
            dists[oids.mask, :] = INVDIST
            dists[:, hids.mask] = INVDIST
        
            rids, cids = linear_sum_assignment(dists)
            for i, j in zip(rids, cids):                
                if dists[i, j] == INVDIST:
                    continue
                
                o = oids[i]
                h = hids.data[j]
                is_switch = o in self.m and \
                            self.m[o] != h and \
                            abs(frameid - self.last_occurrence[o]) <= self.max_switch_time
                cat = 'SWITCH' if is_switch else 'MATCH'
                self.events.loc[(frameid, next(eid)), :] = [cat, oids.data[i], hids.data[j], dists[i, j]]
                oids[i] = ma.masked
                hids[j] = ma.masked
                self.m[o] = h

        # 3. All remaining objects are missed
        for o in oids[~oids.mask]:
            self.events.loc[(frameid, next(eid)), :] = ['MISS', o, np.nan, np.nan]
        
        # 4. All remaining hypotheses are false alarms
        for h in hids[~hids.mask]:
            self.events.loc[(frameid, next(eid)), :] = ['FP', np.nan, h, np.nan]

        # 5. Update occurance state
        for o in oids.data:
            self.last_occurrence[o] = frameid

        if frameid in self.events.index:
            return self.events.loc[frameid]
        else:
            return None

    @staticmethod
    def new_event_dataframe():
        """Create a new DataFrame for event tracking."""
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
    
    def _sanitize_dists(self, dists):
        """Replace invalid distances."""
        
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