"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Toka, 2018
Origin: https://github.com/cheind/py-motmetrics
Extended: <reposity>
"""

import numpy as np
import pandas as pd
from configparser import ConfigParser
from motmetrics.lap import linear_sum_assignment
import motmetrics.distances as mmd
import time
import logging

def preprocessResult(res, gt, inifile):
    st = time.time()
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
    todrop = []
    for t in range(1,F+1):
        if t not in res.index or t not in gt.index: continue
        #st = time.time()
        resInFrame = res.loc[t]
        N = len(resInFrame)

        GTInFrame = gt.loc[t]
        Ngt = len(GTInFrame)
        A = GTInFrame[['X','Y','Width','Height']].values
        B = resInFrame[['X','Y','Width','Height']].values
        disM = mmd.iou_matrix(A, B, max_iou = 0.5)
        #en = time.time()
        #print('----', 'disM', en - st)
        le, ri = linear_sum_assignment(disM)
        flags = [1 if distractors[it['ClassId']] or it['Visibility']<0. else 0 for i,(k,it) in enumerate(GTInFrame.iterrows())]
        hid = [k for k,it in resInFrame.iterrows()]
        for i, j in zip(le, ri):
            if not np.isfinite(disM[i, j]):
                continue
            if flags[i]:
                todrop.append((t, hid[j]))
        #en = time.time()
        #print('Frame %d: '%t, en - st)
    ret = res.drop(labels=todrop)
    logging.info('Preprocess take %.3f seconds and remove %d boxes.'%(time.time() - st, len(todrop)))
    return ret
