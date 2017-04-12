"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

from enum import Enum
import pandas as pd
import numpy as np
import io

class Format(Enum):
    """Enumerates supported file formats."""

    MOT16 = 'mot16'
    """Milan, Anton, et al. "Mot16: A benchmark for multi-object tracking." arXiv preprint arXiv:1603.00831 (2016)."""

    MOT15_2D = 'mot15-2D'
    """Leal-Taixé, Laura, et al. "MOTChallenge 2015: Towards a benchmark for multi-target tracking." arXiv preprint arXiv:1504.01942 (2015)."""

    VATIC_TXT = 'vatic-txt'
    """Vondrick, Carl, Donald Patterson, and Deva Ramanan. "Efficiently scaling up crowdsourced video annotation." International Journal of Computer Vision 101.1 (2013): 184-204.
    https://github.com/cvondrick/vatic
    """


def _load_motchallenge(fname, **kwargs):
    """Load MOT challenge data."""

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
        
    # Account for matlab convention.
    df[['X', 'Y']] -= (1, 1)

    # Removed trailing column
    del df['unused']

    return df

def _load_vatictxt(fname, **kwargs):
    # The vatic .txt format is a variable number of columns CSV format. First then entires are fixed,
    # then a variable number of activities present at the current frame. Therefore we cannot use 
    # pandas CSV import directly.

    
    with open(fname) as f:        
        # First time going over file, we collect the set of all variable activities
        activities = set()
        for line in f:
            [activities.add(c) for c in line.split()[10:]]        
        activitylist = list(activities)

        # Second time we construct artificial binary columns for each activity
        data = []
        f.seek(0)
        for line in f:
            fields = line.split()
            attrs = ['0'] * len(activitylist)            
            for a in fields[10:]:
                 attrs[activitylist.index(a)] = '1'
            fields = fields[:10]
            fields.extend(attrs)
            data.append(' '.join(fields))

        strdata = '\n'.join(data)
        print(strdata)

        dtype = {
            'Id': np.int64,
            'X': np.float32,
            'Y': np.float32,
            'Width': np.float32,
            'Height': np.float32,
            'FrameId': np.int64,
            'Lost': bool,
            'Occluded': bool,
            'Generated': bool,
            'ClassId': str,
        }

        # Remove quotes from activities
        activitylist = [a.replace('\"', '').capitalize() for a in activitylist]        

        for a in activitylist:
            dtype[a] = bool
    
        names = ['Id', 'X', 'Y', 'Width', 'Height', 'FrameId', 'Lost', 'Occluded', 'Generated', 'ClassId']
        names.extend(activitylist)
        return pd.read_csv(io.StringIO(strdata), names=names, index_col=['FrameId','Id'], header=None, sep=' ')


    """
     # Cannot use read_csv from pandas directly
    with open(fname) as f:
        lines = f.readlines()

    # Each line:
    # 0   Track ID. All rows with the same ID belong to the same path.
    # 1   xmin. The top left x-coordinate of the bounding box.
    # 2   ymin. The top left y-coordinate of the bounding box.
    # 3   xmax. The bottom right x-coordinate of the bounding box.
    # 4   ymax. The bottom right y-coordinate of the bounding box.
    # 5   frame. The frame that this annotation represents.
    # 6   lost. If 1, the annotation is outside of the view screen.
    # 7   occluded. If 1, the annotation is occluded.
    # 8   generated. If 1, the annotation was automatically interpolated.
    # 9  label. The label for this annotation, enclosed in quotation marks.
    # 10+ attributes. Each column after this is an attribute.

    newlines = []
    for l in lines:
        cols = [c.replace('\"', '') for c in l.split()]
        # Map attributes to separate columns
        attrs = ['0'] * len(attributes)
        for a in cols[10:]:
            try:
                idx = attributes.index(a)
                attrs[idx] = '1'
            except ValueError:
                pass
        
        cols = cols[:10]
        cols.extend(attrs)
        newlines.append(' '.join(cols))

    f = '\n'.join(newlines)

    dtype = {
        'id': np.int64,
        'xmin': np.float32,
        'ymin': np.float32,
        'xmax': np.float32,
        'ymax': np.float32,
        'frame': np.int64,
        'lost': bool,
        'occluded': bool,
        'generated': bool,
        'label': str,
    }

    for a in attributes:
        dtype[a] = bool
    
    header = ['id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label']
    header.extend(attributes)
    return pd.read_csv(io.StringIO(f), names=header, header=None, dtype=dtype, sep=' ')

"""
    #raise NotImplementedError()

def loadtxt(fname, fmt='mot15-2D', **kwargs):
    """Load data from any known format."""
    fmt = Format(fmt)

    switcher = {
        Format.MOT16: _load_motchallenge,
        Format.MOT15_2D: _load_motchallenge,
        Format.VATIC_TXT: _load_vatictxt
    }
    func = switcher.get(fmt)
    return func(fname, **kwargs)

def render_summary(summary, formatters=None, namemap=None, buf=None):
    """Render metrics summary to console friendly tabular output.
    
    Params
    ------
    summary : pd.DataFrame
        Dataframe containing summaries in rows.    
    
    Kwargs
    ------
    buf : StringIO-like, optional
        Buffer to write to
    formatters : dict, optional
        Dicionary defining custom formatters for individual metrics.
        I.e `{'mota': '{:.2%}'.format}`. You can get preset formatters
        from MetricsHost.formatters
    namemap : dict, optional
        Dictionary defining new metric names for display. I.e 
        `{'num_false_positives': 'FP'}`.

    Returns
    -------
    string
        Formatted string
    """

    if not namemap is None:
        summary = summary.rename(columns=namemap)
        if not formatters is None:
            formatters = dict([(namemap[c], f) if c in namemap else (c, f) for c, f in formatters.items()])

    output = summary.to_string(
        buf=buf,
        formatters=formatters,
    )

    return output

motchallenge_metric_names = {
    'recall' : 'Rcll', 
    'precision' : 'Prcn',
    'num_unique_objects' : 'GT', 
    'mostly_tracked' : 'MT', 
    'partially_tracked' : 'PT', 
    'mostly_lost': 'ML',  
    'num_false_positives' : 'FP', 
    'num_misses' : 'FN',
    'num_switches' : 'IDs',
    'num_fragmentations' : 'FM',
    'mota' : 'MOTA',
    'motp' : 'MOTP'
}
"""A list mappings for metric names to comply with MOTChallenge."""
