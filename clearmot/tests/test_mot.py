from pytest import approx
import numpy as np
import pandas as pd
import clearmot as cm

def test_events():
    acc = cm.new_accumulator()

    # All FP
    cm.update_mot(acc, [], ['a', 'b'], [], frameid=0)
    # All miss
    cm.update_mot(acc, [1, 2], [], [], frameid=1)
    # Match
    cm.update_mot(acc, [1, 2], ['a', 'b'], [[1, 0.5], [0.3, 1]], frameid=2)
    # Switch
    cm.update_mot(acc, [1, 2], ['a', 'b'], [[0.2, np.nan], [np.nan, 0.1]], frameid=3)
    # Match. Better new match is available but should prefer history
    cm.update_mot(acc, [1, 2], ['a', 'b'], [[5, 1], [1, 5]], frameid=4)
    # No data
    cm.update_mot(acc, [], [], [], frameid=5)

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
    assert cm.metrics.MOTP(acc) == approx(11.1 / 6)
    assert cm.metrics.MOTA(acc) == approx(1. - (2 + 2 + 2) / 8)


def test_correct_average():
    # Tests what is being depicted in figure 3 of 'Evaluating MOT Performance'
    acc = cm.new_accumulator(auto_id=True)
    
    # No track
    cm.update_mot(acc, [1, 2, 3, 4], [], [])
    cm.update_mot(acc, [1, 2, 3, 4], [], [])
    cm.update_mot(acc, [1, 2, 3, 4], [], [])
    cm.update_mot(acc, [1, 2, 3, 4], [], [])

    # Track single
    cm.update_mot(acc, [4], [4], [0])
    cm.update_mot(acc, [4], [4], [0])
    cm.update_mot(acc, [4], [4], [0])
    cm.update_mot(acc, [4], [4], [0])

    assert cm.metrics.MOTA(acc) == approx(0.2)