"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

import pandas as pd
import numpy as np

from .mot import MOTAccumulator
from .distances import iou_matrix, norm2squared_matrix

def compare_to_groundtruth(gt, dt, dist='iou', distfields=['X', 'Y', 'Width', 'Height'], distth=0.5):
    """Compare groundtruth and detector results.

    This method assumes both results are given in terms of DataFrames with at least the following fields
     - `FrameId` First level index used for matching ground-truth and test frames.
     - `Id` Secondary level index marking available object / hypothesis ids
    
    Depending on the distance to be used relevant distfields need to be specified.

    Params
    ------
    gt : pd.DataFrame
        Dataframe for ground-truth
    test : pd.DataFrame
        Dataframe for detector results
    
    Kwargs
    ------
    dist : str, optional
        String identifying distance to be used. Defaults to intersection over union.
    distfields: array, optional
        Fields relevant for extracting distance information. Defaults to ['X', 'Y', 'Width', 'Height']
    distth: float, optional
        Maximum tolerable distance. Pairs exceeding this threshold are marked 'do-not-pair'.
    """

    acc = MOTAccumulator()

    for frameid, fgt in gt.groupby(level=0):
        fgt = fgt.loc[frameid] 
        oids = fgt.index.values
        
        hids = None
        dists = None

        if frameid in dt.index:
            fdt = dt.loc[frameid]
            hids = fdt.index.values
            if dist == 'iou':                
                dists = iou_matrix(fgt[distfields].values, fdt[distfields].values, max_iou=distth)
            else:
                dists = norm2squared_matrix(fgt[distfields].values, fdt[distfields].values, max_d2=distth)

        acc.update(oids, hids, dists, frameid=frameid)

    return acc