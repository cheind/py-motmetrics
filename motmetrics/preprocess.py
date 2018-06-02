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
