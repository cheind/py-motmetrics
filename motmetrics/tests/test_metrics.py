from pytest import approx
import numpy as np
import pandas as pd
import motmetrics as mm
import pytest
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')

def test_metricscontainer_1():
    m = mm.metrics.MetricsHost()
    m.register(lambda df: 1., name='a')
    m.register(lambda df: 2., name='b')
    m.register(lambda df, a, b: a+b, deps=['a', 'b'], name='add')
    m.register(lambda df, a, b: a-b, deps=['a', 'b'], name='sub')
    m.register(lambda df, a, b: a*b, deps=['add', 'sub'], name='mul')
    summary = m.compute(mm.MOTAccumulator.new_event_dataframe(), metrics=['mul','add'], name='x')
    assert summary.columns.values.tolist() == ['mul','add']
    assert summary.iloc[0]['mul'] == -3.
    assert summary.iloc[0]['add'] == 3.

def test_metricscontainer_autodep():
    m = mm.metrics.MetricsHost()
    m.register(lambda df: 1., name='a')
    m.register(lambda df: 2., name='b')
    m.register(lambda df, a, b: a+b, name='add', deps='auto')
    m.register(lambda df, a, b: a-b, name='sub', deps='auto')
    m.register(lambda df, add, sub: add*sub, name='mul', deps='auto')
    summary = m.compute(mm.MOTAccumulator.new_event_dataframe(), metrics=['mul','add'])
    assert summary.columns.values.tolist() == ['mul','add']
    assert summary.iloc[0]['mul'] == -3.
    assert summary.iloc[0]['add'] == 3.

def test_metricscontainer_autoname():

    def constant_a(df):
        """Constant a help."""
        return 1.
    
    def constant_b(df):
        return 2.

    def add(df, constant_a, constant_b):
        return constant_a + constant_b

    def sub(df, constant_a, constant_b):
        return constant_a - constant_b

    def mul(df, add, sub):
        return add * sub

    m = mm.metrics.MetricsHost()
    m.register(constant_a, deps='auto')
    m.register(constant_b, deps='auto')
    m.register(add, deps='auto')
    m.register(sub, deps='auto')
    m.register(mul, deps='auto')

    assert m.metrics['constant_a']['help'] == 'Constant a help.'

    summary = m.compute(mm.MOTAccumulator.new_event_dataframe(), metrics=['mul','add'])
    assert summary.columns.values.tolist() == ['mul','add']
    assert summary.iloc[0]['mul'] == -3.
    assert summary.iloc[0]['add'] == 3.

def test_mota_motp():
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
    
    mh = mm.metrics.create()
    metr = mh.compute(acc, metrics=['motp', 'mota', 'num_predictions'], return_dataframe=False, return_cached=True)

    assert metr['num_matches'] == 4
    assert metr['num_false_positives'] == 2
    assert metr['num_misses'] == 2
    assert metr['num_switches'] == 2
    assert metr['num_detections'] == 6
    assert metr['num_objects'] == 8
    assert metr['num_predictions'] == 8
    assert metr['mota'] == approx(1. - (2 + 2 + 2) / 8)
    assert metr['motp'] == approx(11.1 / 6)
    

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

def test_motchallenge_files():
    dnames = [
        'TUD-Campus',
        'TUD-Stadtmitte',
    ]
    
    def compute_motchallenge(dname):
        df_gt = mm.io.loadtxt(os.path.join(dname,'gt.txt'))
        df_test = mm.io.loadtxt(os.path.join(dname,'test.txt'))
        return mm.utils.compare_to_groundtruth(df_gt, df_test, 'iou', distth=0.5)

    accs = [compute_motchallenge(os.path.join(DATA_DIR, d)) for d in dnames]

    # For testing
    # [a.events.to_pickle(n) for (a,n) in zip(accs, dnames)]

    mh = mm.metrics.create()
    summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, names=dnames, generate_overall=True)

    print()
    print(mm.io.render_summary(summary, namemap=mm.io.motchallenge_metric_names, formatters=mh.formatters))

    expected = pd.DataFrame([
        [0.557659, 0.729730, 0.451253, 0.582173, 0.941441, 8.0, 1, 6, 1, 13, 150, 7, 7, 0.526462, 0.277201],
        [0.644619, 0.819760, 0.531142, 0.608997, 0.939920, 10.0, 5, 4, 1, 45, 452, 7, 6, 0.564014, 0.345904],
        [0.624296, 0.799176, 0.512211, 0.602640, 0.940268, 18.0, 6, 10, 2, 58, 602, 14, 13, 0.555116, 0.330177],
    ])

    np.testing.assert_allclose(summary, expected, atol=1e-3)