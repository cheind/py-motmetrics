"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Toka, 2018
Origin: https://github.com/cheind/py-motmetrics
Extended: <reposity>
"""

import numpy as np
import numpy.ma as ma
import pandas as pd
from collections import OrderedDict
from configparser import ConfigParser
from itertools import count
from motmetrics.lap import linear_sum_assignment
import motmetrics.distances as mmd


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
    
    self.dirty_events = True
    oids = ma.array(oids, mask=np.zeros(len(oids)))
    hids = ma.array(hids, mask=np.zeros(len(hids)))  
    dists = np.atleast_2d(dists).astype(float).reshape(oids.shape[0], hids.shape[0]).copy()

    if frameid is None:            
        assert self.auto_id, 'auto-id is not enabled'
        if len(self._indices) > 0:
            frameid = self._indices[-1][0] + 1
        else:
            frameid = 0
    else:
        assert not self.auto_id, 'Cannot provide frame id when auto-id is enabled'
    
    eid = count()

    # 0. Record raw events

    no = len(oids)
    nh = len(hids)
    
    if no * nh > 0:
        for i in range(no):
            for j in range(nh):
                self._indices.append((frameid, next(eid)))
                self._events.append(['RAW', oids[i], hids[j], dists[i,j]])
    elif no == 0:
        for i in range(nh):
            self._indices.append((frameid, next(eid)))
            self._events.append(['RAW', np.nan, hids[i], np.nan])       
    elif nh == 0:
        for i in range(no):
            self._indices.append((frameid, next(eid)))
            self._events.append(['RAW', oids[i], np.nan, np.nan])

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

            if np.isfinite(dists[i,j]):
                oids[i] = ma.masked
                hids[j] = ma.masked
                self.m[oids.data[i]] = hids.data[j]
                
                self._indices.append((frameid, next(eid)))
                self._events.append(['MATCH', oids.data[i], hids.data[j], dists[i, j]])

        # 2. Try to remaining objects/hypotheses
        dists[oids.mask, :] = np.nan
        dists[:, hids.mask] = np.nan
    
        rids, cids = linear_sum_assignment(dists)

        for i, j in zip(rids, cids):                
            if not np.isfinite(dists[i,j]):
                continue
            
            o = oids[i]
            h = hids.data[j]
            is_switch = o in self.m and \
                        self.m[o] != h and \
                        abs(frameid - self.last_occurrence[o]) <= self.max_switch_time
            cat = 'SWITCH' if is_switch else 'MATCH'
            self._indices.append((frameid, next(eid)))
            self._events.append([cat, oids.data[i], hids.data[j], dists[i, j]])
            oids[i] = ma.masked
            hids[j] = ma.masked
            self.m[o] = h

    # 3. All remaining objects are missed
    for o in oids[~oids.mask]:
        self._indices.append((frameid, next(eid)))
        self._events.append(['MISS', o, np.nan, np.nan])
    
    # 4. All remaining hypotheses are false alarms
    for h in hids[~hids.mask]:
        self._indices.append((frameid, next(eid)))
        self._events.append(['FP', np.nan, h, np.nan])

    # 5. Update occurance state
    for o in oids.data:            
        self.last_occurrence[o] = frameid

    return frameid

def boxiou(a, b):
    rx1 = max(a[0], b[0])
    rx2 = min(a[0]+a[2], b[0]+b[2])
    ry1 = max(a[1], b[1])
    ry2 = min(a[1]+a[3], b[1]+b[3])
    if ry2>ry1 and rx2>rx1:
        i = (ry2-ry1)*(rx2-rx1)
        u = x.area()+y.area()-i
        return float(i)/u
    else: return 0.0

def preprocessResult(res, gt, inifile):
    labels = ['ped',           # 1 
    'person_on_vhcl',    # 2 
    'car',               # 3 
    'bicycle',           # 4 
    'mbike',             # 5 
    'non_mot_vhcl',      # 6 
    'static_person',     # 7 
    'distractor',        # 8 
    'occluder',          # 9 
    'occluder_on_grnd',      #10 
    'occluder_full',         # 11
    'reflection',        # 12
    'crowd'          # 13
    ] 
    distractors_ = ['person_on_vhcl','static_person','distractor','reflection']
    distractors = {i+1 : x in distractors_ for i,x in enumerate(labels)}
    for i in distractors_:
        distractors[i] = 1
    seqIni = ConfigParser()
    seqIni.read(inifile, encoding='utf8')
    F = int(seqIni['Sequence']['seqLength'])
    for t in range(1,F+1):
        resInFrame = res.loc[t]
        N = len(resInFrame)

        GTInFrame = gt.loc[t]
        Ngt = len(GTInFrame)
        gtb = []
        dtb = []
        for i in range(len(GTInFrame)):
            bgt = \
                (GTInFrame.iloc[i]['X'],
                 GTInFrame.iloc[i]['Y'],
                 GTInFrame.iloc[i]['Width'],
                 GTInFrame.iloc[i]['Height']
                )
            gtb.append(bgt)
        for j in range(len(resInFrame)):
            bres = \
              (resInFrame.iloc[j]['X'],
               resInFrame.iloc[j]['Y'],
               resInFrame.iloc[j]['Width'],
               resInFrame.iloc[j]['Height']
              )
            dtb.append(bres)
        A = np.array(gtb)
        B = np.array(dtb)
        disM = mmd.iou_matrix(A, B, max_iou = 0.5)
        le, ri = linear_sum_assignment(disM)
        #print('-'*20)
        for i, j in zip(le, ri):
            if not np.isfinite(disM[i, j]):
                continue
            #print(i, j, disM[i, j])
            gtbox = GTInFrame.iloc[i]
            resbox = resInFrame.iloc[j]
            clsid = gtbox['ClassId']
            #print(clsid, distractors[clsid], distractors[int(clsid)])
            if distractors[clsid] or gtbox['Visibility']<0.:
                res.drop(labels=(t, resbox.name), inplace=True)
    return res
