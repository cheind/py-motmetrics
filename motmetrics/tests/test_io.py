# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Tests IO functions."""

import os
from io import StringIO

import numpy as np

import motmetrics._io as io

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')


def test_infer_format():
    """Tests format inference from extension and text contents."""
    assert io.infer_format(os.path.join(DATA_DIR, 'iotest/motchallenge.txt')) == io.Format.MOT15_2D
    assert io.infer_format(os.path.join(DATA_DIR, 'iotest/vatic.txt')) == io.Format.VATIC_TXT
    assert io.infer_format(os.path.join(DATA_DIR, 'iotest/detrac.mat')) == io.Format.DETRAC_MAT
    assert io.infer_format(os.path.join(DATA_DIR, 'iotest/detrac.xml')) == io.Format.DETRAC_XML


def test_loadtxt_auto():
    """Tests AUTO format dispatch matches explicit formats."""
    cases = [
        ('iotest/motchallenge.txt', io.Format.MOT15_2D),
        ('iotest/vatic.txt', io.Format.VATIC_TXT),
        ('iotest/detrac.mat', io.Format.DETRAC_MAT),
        ('iotest/detrac.xml', io.Format.DETRAC_XML),
    ]
    for filename, fmt in cases:
        path = os.path.join(DATA_DIR, filename)
        expected = io.loadtxt(path, fmt=fmt)
        actual = io.loadtxt(path, fmt='auto')
        _assert_sequence_equal(actual, expected)


def test_load_vatic():
    """Tests VATIC_TXT format."""
    data = io.loadtxt(os.path.join(DATA_DIR, 'iotest/vatic.txt'), fmt=io.Format.VATIC_TXT)

    expected = np.asarray([
        # F,ID,Y,W,H,L,O,G,F,A1,A2,A3,A4
        (0, 0, 412, 0, 430, 124, 0, 0, 0, 'worker', 0, 0, 0, 0),
        (1, 0, 412, 10, 430, 114, 0, 0, 1, 'pc', 1, 0, 1, 0),
        (1, 1, 412, 0, 430, 124, 0, 0, 1, 'pc', 0, 1, 0, 0),
        (2, 2, 412, 0, 430, 124, 0, 0, 1, 'worker', 1, 1, 0, 1)
    ], dtype=object)

    actual = _sequence_rows(
        data,
        ['X', 'Y', 'Width', 'Height', 'Lost', 'Occluded', 'Generated', 'ClassId', 'Attr1', 'Attr2', 'Attr3', 'Attr4'],
    )
    np.testing.assert_equal(actual, expected)


def test_load_motchallenge():
    """Tests MOT15_2D format."""
    data = io.loadtxt(os.path.join(DATA_DIR, 'iotest/motchallenge.txt'), fmt=io.Format.MOT15_2D)

    expected = np.asarray([
        (1, 1, 398, 181, 121, 229, 1, -1, -1),  # Note -1 on x and y for correcting matlab
        (1, 2, 281, 200, 92, 184, 1, -1, -1),
        (2, 2, 268, 201, 87, 182, 1, -1, -1),
        (2, 3, 70, 150, 100, 284, 1, -1, -1),
        (2, 4, 199, 205, 55, 137, 1, -1, -1),
    ], dtype=float)

    np.testing.assert_equal(
        _sequence_rows(data, ['X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility']).astype(float),
        expected,
    )


def test_load_motchallenge_infers_whitespace_separator(tmp_path):
    """Tests fast loading of whitespace-delimited MOTChallenge data."""
    path = tmp_path / 'motchallenge.txt'
    path.write_text('1 7 11 21 30 40 1 -1 -1 -1\n', encoding='utf-8')

    data = io.load_motchallenge(path)

    assert data.values(['X', 'Y', 'Width', 'Height'])[0].tolist() == [10, 20, 30, 40]


def test_load_motchallenge_infers_separator_for_file_object():
    """Tests separator inference without consuming a caller-owned stream."""
    source = StringIO('1,7,11,21,30,40,1,-1,-1,-1\n')

    data = io.load_motchallenge(source)

    assert data.values(['X', 'Y', 'Width', 'Height'])[0].tolist() == [10, 20, 30, 40]


def test_load_detrac_mat():
    """Tests DETRAC_MAT format."""
    data = io.loadtxt(os.path.join(DATA_DIR, 'iotest/detrac.mat'), fmt=io.Format.DETRAC_MAT)

    expected = np.asarray([
        (1., 1., 745., 356., 148., 115., 1., -1., -1.),
        (2., 1., 738., 350., 145., 111., 1., -1., -1.),
        (3., 1., 732., 343., 142., 107., 1., -1., -1.),
        (4., 1., 725., 336., 139., 104., 1., -1., -1.)
    ])

    np.testing.assert_equal(
        _sequence_rows(data, ['X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility']).astype(float),
        expected,
    )


def test_load_detrac_xml():
    """Tests DETRAC_XML format."""
    data = io.loadtxt(os.path.join(DATA_DIR, 'iotest/detrac.xml'), fmt=io.Format.DETRAC_XML)

    expected = np.asarray([
        (1., 1., 744.6, 356.33, 148.2, 115.14, 1., -1., -1.),
        (2., 1., 738.2, 349.51, 145.21, 111.29, 1., -1., -1.),
        (3., 1., 731.8, 342.68, 142.23, 107.45, 1., -1., -1.),
        (4., 1., 725.4, 335.85, 139.24, 103.62, 1., -1., -1.)
    ])

    np.testing.assert_allclose(
        _sequence_rows(data, ['X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility']).astype(float),
        expected,
    )


def _sequence_rows(data, fields):
    return np.column_stack((data.frame_ids, data.ids, *(data.column(field) for field in fields)))


def _assert_sequence_equal(actual, expected):
    np.testing.assert_equal(actual.frame_ids, expected.frame_ids)
    np.testing.assert_equal(actual.ids, expected.ids)
    assert actual._fields.keys() == expected._fields.keys()
    for field in actual._fields:
        np.testing.assert_equal(actual.column(field), expected.column(field))
