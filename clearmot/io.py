"""CLEAR MOT - metrics for multiple object tracker evaluation.

This library provides CLEAR MOT metrics for multiple object tracker evaluation
in Python. The evaluation process is distance metric agnostic making it applicable
to various scenearios (centroid distance, intersection over union and more).

Christoph Heindl, 2017
https://github.com/cheind/py-clearmot
"""

from enum import Enum
import pandas as pd

class Format(Enum):
    """Enumerates supported file formats."""

    MOT16 = 'mot16'
    """Milan, Anton, et al. "Mot16: A benchmark for multi-object tracking." arXiv preprint arXiv:1603.00831 (2016)."""

    MOT152D = 'mot15-2D'
    """Leal-Taixé, Laura, et al. "MOTChallenge 2015: Towards a benchmark for multi-target tracking." arXiv preprint arXiv:1504.01942 (2015)."""

def _load_motchallenge(fname, **kwargs):
    sep = kwargs.pop('sep', '\s+|\t+|,')
    df = pd.read_csv(
        fname, 
        sep=sep, 
        index_col=[0,1], 
        skipinitialspace=True, 
        header=None,
        names=['FrameId', 'Id', 'x', 'y', 'w', 'h', 'Score', 'ClassId', 'Visibility', 'unused'],
        engine='python'
    )

    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['w'] = df['w'].astype(float)
    df['h'] = df['h'].astype(float)

    del df['unused']
    return df

def loadtxt(fname, fmt='mot16', **kwargs):
    fmt = Format(fmt)

    switcher = {
        Format.MOT16: _load_motchallenge,
        Format.MOT152D: _load_motchallenge
    }
    func = switcher.get(fmt)
    return func(fname, **kwargs)





