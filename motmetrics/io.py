"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

from enum import Enum
import pandas as pd
import numpy as np

class Format(Enum):
    """Enumerates supported file formats."""

    MOT16 = 'mot16'
    """Milan, Anton, et al. "Mot16: A benchmark for multi-object tracking." arXiv preprint arXiv:1603.00831 (2016)."""

    MOT15_2D = 'mot15-2D'
    """Leal-Taixé, Laura, et al. "MOTChallenge 2015: Towards a benchmark for multi-target tracking." arXiv preprint arXiv:1504.01942 (2015)."""

def _load_motchallenge(fname, **kwargs):
    sep = kwargs.pop('sep', '\s+|\t+|,')
    df = pd.read_csv(
        fname, 
        sep=sep, 
        index_col=[0,1], 
        skipinitialspace=True, 
        header=None,
        names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Score', 'ClassId', 'Visibility', 'unused'],
        engine='python'
    )
        
    # Account for matlab convention start is (1, 1)
    df[['X', 'Y']] -= (1, 1)

    # Removed trailing column
    del df['unused']

    return df

def loadtxt(fname, fmt='mot15-2D', **kwargs):
    fmt = Format(fmt)

    switcher = {
        Format.MOT16: _load_motchallenge,
        Format.MOT15_2D: _load_motchallenge
    }
    func = switcher.get(fmt)
    return func(fname, **kwargs)





