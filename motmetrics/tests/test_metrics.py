from pytest import approx
import numpy as np
import pandas as pd
import motmetrics as mm
import pytest

def test_metrics_host():
    m = mm.metrics.Metrics()
    m.add(lambda df: 1., name='a')
    m.add(lambda df: 2., name='b')
    m.add(lambda df, a, b: a+b, deps=['a', 'b'], name='add')
    m.add(lambda df, a, b: a-b, deps=['a', 'b'], name='sub')
    m.add(lambda df, a, b: a*b, deps=['add', 'sub'], name='mul')
    summary = m.summarize(None, metrics=['mul','add'])
    assert 'mul' in summary
    assert 'add' in summary
    assert not 'sub' in summary
    assert not 'a' in summary
    assert not 'b' in summary
    assert summary['mul'] == -3.
    assert summary['add'] == 3.

def test_metrics_host_autodep():
    m = mm.metrics.Metrics()
    m.add(lambda df: 1., name='a')
    m.add(lambda df: 2., name='b')
    m.add(lambda df, a, b: a+b, name='add', deps='auto')
    m.add(lambda df, a, b: a-b, name='sub', deps='auto')
    m.add(lambda df, add, sub: add*sub, name='mul', deps='auto')
    summary = m.summarize(None, metrics=['mul','add'])
    assert 'mul' in summary
    assert 'add' in summary
    assert not 'sub' in summary
    assert not 'a' in summary
    assert not 'b' in summary
    assert summary['mul'] == -3.
    assert summary['add'] == 3.

def test_metrics_host_autodep_autoname():

    def constant_a(df):
        return 1.
    
    def constant_b(df):
        return 2.

    def add(df, constant_a, constant_b):
        return constant_a + constant_b

    def sub(df, constant_a, constant_b):
        return constant_a - constant_b

    def mul(df, add, sub):
        return add * sub

    m = mm.metrics.Metrics()
    m.add(constant_a, deps='auto')
    m.add(constant_b, deps='auto')
    m.add(add, deps='auto')
    m.add(sub, deps='auto')
    m.add(mul, deps='auto')
    summary = m.summarize(None, metrics=['mul','add'])
    assert 'mul' in summary
    assert 'add' in summary
    assert not 'sub' in summary
    assert not 'a' in summary
    assert not 'b' in summary
    assert summary['mul'] == -3.
    assert summary['add'] == 3.
