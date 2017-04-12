from pytest import approx
import numpy as np
import pandas as pd
import motmetrics as mm
import os
import pytest

def test_events():
    acc = mm.MOTAccumulator()

    # All FP
    acc.update([], ['a', 'b'], [], frameid=0)
    # All miss
    acc.update([1, 2], [], [], frameid=1)
    # Match
    acc.update([1, 2], ['a', 'b'], [[1, 0.5], [0.3, 1]], frameid=2)
    # Switch
    acc.update([1, 2], ['a', 'b'], [[0.2, np.nan], [np.nan, 0.1]], frameid=3)
    # Match. Better new match is available but should prefer history
    acc.update([1, 2], ['a', 'b'], [[5, 1], [1, 5]], frameid=4)
    # No data
    acc.update([], [], [], frameid=5)

    expect = mm.MOTAccumulator.new_event_dataframe()
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
    
    mh = mm.metrics.create()
    metr = mh.compute(acc, metrics=['motp', 'mota'], return_dataframe=False)
    assert metr['motp'] == approx(11.1 / 6)
    assert metr['mota'] == approx(1. - (2 + 2 + 2) / 8)

def test_max_switch_time():
     acc = mm.MOTAccumulator(max_switch_time=1)
     df = acc.update([1, 2], ['a', 'b'], [[1, 0.5], [0.3, 1]], frameid=1) # 1->a, 2->b
     df = acc.update([1, 2], ['a', 'b'], [[0.5, np.nan], [np.nan, 0.5]], frameid=2) # 1->b, 2->a 
     assert (df.Type == 'SWITCH').all()

     acc = mm.MOTAccumulator(max_switch_time=1)
     df = acc.update([1, 2], ['a', 'b'], [[1, 0.5], [0.3, 1]], frameid=1) # 1->a, 2->b
     df = acc.update([1, 2], ['a', 'b'], [[0.5, np.nan], [np.nan, 0.5]], frameid=5) # Later frame 1->b, 2->a 
     assert (df.Type == 'MATCH').all()

def test_auto_id():
    acc = mm.MOTAccumulator(auto_id=True)
    acc.update([1, 2, 3, 4], [], [])
    acc.update([1, 2, 3, 4], [], [])
    assert acc.events.index.levels[0][-1] == 1
    acc.update([1, 2, 3, 4], [], [])
    assert acc.events.index.levels[0][-1] == 2

    with pytest.raises(AssertionError):
        acc.update([1, 2, 3, 4], [], [], frameid=5)
    
    acc = mm.MOTAccumulator(auto_id=False)
    with pytest.raises(AssertionError):
        acc.update([1, 2, 3, 4], [], [])

def test_correct_average():
    # Tests what is being depicted in figure 3 of 'Evaluating MOT Performance'
    acc = mm.MOTAccumulator(auto_id=True)

    # No track
    acc.update([1, 2, 3, 4], [], [])
    acc.update([1, 2, 3, 4], [], [])
    acc.update([1, 2, 3, 4], [], [])
    acc.update([1, 2, 3, 4], [], [])

    # Track single
    acc.update([4], [4], [0])
    acc.update([4], [4], [0])
    acc.update([4], [4], [0])
    acc.update([4], [4], [0])

    mh = mm.metrics.create()
    metr = mh.compute(acc, metrics='mota', return_dataframe=False)
    assert metr['mota'] == approx(0.2)