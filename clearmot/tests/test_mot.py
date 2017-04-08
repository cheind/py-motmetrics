from pytest import approx
import numpy as np
import pandas as pd
import clearmot as cm
import os

def test_events():
    acc = cm.new_accumulator()

    # All FP
    cm.update(acc, [], ['a', 'b'], [], frameid=0)
    # All miss
    cm.update(acc, [1, 2], [], [], frameid=1)
    # Match
    cm.update(acc, [1, 2], ['a', 'b'], [[1, 0.5], [0.3, 1]], frameid=2)
    # Switch
    cm.update(acc, [1, 2], ['a', 'b'], [[0.2, np.nan], [np.nan, 0.1]], frameid=3)
    # Match. Better new match is available but should prefer history
    cm.update(acc, [1, 2], ['a', 'b'], [[5, 1], [1, 5]], frameid=4)
    # No data
    cm.update(acc, [], [], [], frameid=5)

    expect = cm.new_dataframe()
    expect.loc[(0, 0), :] = ['FP', np.nan, 'a', np.nan]
    expect.loc[(0, 1), :] = ['FP', np.nan, 'b', np.nan]
    expect.loc[(1, 0), :] = ['MISS', 1, np.nan, np.nan]
    expect.loc[(1, 1), :] = ['MISS', 2, np.nan, np.nan]
    expect.loc[(2, 0), :] = ['MATCH', 1, 'b', 0.5]
    expect.loc[(2, 1), :] = ['MATCH', 2, 'a', 0.3]
    expect.loc[(3, 0), :] = ['SWITCH', 1, 'a', 0.2]
    expect.loc[(3, 1), :] = ['SWITCH', 2, 'b', 0.1]
    expect.loc[(4, 0), :] = ['MATCH', 1, 'a', 5.]
    expect.loc[(4, 1), :] = ['MATCH', 2, 'b', 5.]
    # frame 5 generates no events

    assert pd.DataFrame.equals(acc.events, expect)
    
    metr = cm.metrics.compute_metrics(acc)
    assert metr['MOTP'] == approx(11.1 / 6)
    assert metr['MOTA'] == approx(1. - (2 + 2 + 2) / 8)


def test_correct_average():
    # Tests what is being depicted in figure 3 of 'Evaluating MOT Performance'
    acc = cm.new_accumulator(auto_id=True)
    
    # No track
    cm.update(acc, [1, 2, 3, 4], [], [])
    cm.update(acc, [1, 2, 3, 4], [], [])
    cm.update(acc, [1, 2, 3, 4], [], [])
    cm.update(acc, [1, 2, 3, 4], [], [])

    # Track single
    cm.update(acc, [4], [4], [0])
    cm.update(acc, [4], [4], [0])
    cm.update(acc, [4], [4], [0])
    cm.update(acc, [4], [4], [0])

    metr = cm.metrics.compute_metrics(acc)
    assert metr['MOTA'] == approx(0.2)

def compute_motchallenge(dname):

    df_gt = cm.io.loadtxt(os.path.join(dname,'gt.txt'))
    df_test = cm.io.loadtxt(os.path.join(dname,'test.txt'))

    acc = cm.new_accumulator()

    for frameid, dff_gt in df_gt.groupby(level=0):
        dff_gt = dff_gt.loc[frameid]
        if frameid in df_test.index:
            dff_test = df_test.loc[frameid]

            hids = dff_test.index.values
            oids = dff_gt.index.values

            hrects = dff_test[['x', 'y', 'w', 'h']].values
            orects = dff_gt[['x', 'y', 'w', 'h']].values

            dists = cm.distances.iou_matrix(orects, hrects, max_iou=0.5)
            cm.update(acc, oids, hids, dists, frameid=frameid)

    return acc


def test_motchallenge_files():
    dnames = [
        'TUD-Campus',
        'TUD-Stadtmitte',
    ]
    reldir = os.path.join(os.path.dirname(__file__), '../../etc/data')
    accs = [compute_motchallenge(os.path.join(reldir, d)) for d in dnames]
    df = cm.metrics.summarize(accs, dnames)    

    expected = pd.DataFrame([
        [372, 202, 7, 13, 150, 0.277, 0.526, 0.941, 0.582, 7, 8, 1, 6, 1],
        [1201, 697, 7, 45, 452, 0.345, 0.564, 0.939, 0.608, 6, 10, 5, 4, 1],
    ])

    print(df)
    print(expected)
    np.testing.assert_allclose(df, expected, atol=1e-3)