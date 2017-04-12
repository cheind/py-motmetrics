from pytest import approx
import numpy as np
import pandas as pd
import motmetrics as mm
import pytest

def test_metricscontainer_1():
    m = mm.metrics.MetricsContainer()
    m.register(lambda df: 1., name='a')
    m.register(lambda df: 2., name='b')
    m.register(lambda df, a, b: a+b, deps=['a', 'b'], name='add')
    m.register(lambda df, a, b: a-b, deps=['a', 'b'], name='sub')
    m.register(lambda df, a, b: a*b, deps=['add', 'sub'], name='mul')
    summary = m.compute(None, metrics=['mul','add'], name='x')
    assert summary.columns.values.tolist() == ['mul','add']
    assert summary.iloc[0]['mul'] == -3.
    assert summary.iloc[0]['add'] == 3.

def test_metricscontainer_autodep():
    m = mm.metrics.MetricsContainer()
    m.register(lambda df: 1., name='a')
    m.register(lambda df: 2., name='b')
    m.register(lambda df, a, b: a+b, name='add', deps='auto')
    m.register(lambda df, a, b: a-b, name='sub', deps='auto')
    m.register(lambda df, add, sub: add*sub, name='mul', deps='auto')
    summary = m.compute(None, metrics=['mul','add'])
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

    m = mm.metrics.MetricsContainer()
    m.register(constant_a, deps='auto')
    m.register(constant_b, deps='auto')
    m.register(add, deps='auto')
    m.register(sub, deps='auto')
    m.register(mul, deps='auto')

    assert m.metrics['constant_a']['help'] == 'Constant a help.'

    summary = m.compute(None, metrics=['mul','add'])
    assert summary.columns.values.tolist() == ['mul','add']
    assert summary.iloc[0]['mul'] == -3.
    assert summary.iloc[0]['add'] == 3.
