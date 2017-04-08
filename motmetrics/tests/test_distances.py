from pytest import approx
import numpy as np

import motmetrics as mm

def test_norm2squared():
    a = np.array([
        [1., 2],
        [2., 2],
        [3., 2],
    ])

    b = np.array([
        [0., 0],
        [1., 1],      
    ])

    C = mm.distances.norm2squared_matrix(a, b)
    np.testing.assert_allclose(
        C,
        [
            [5, 1],
            [8, 2],
            [13, 5]
        ]
    )

    C = mm.distances.norm2squared_matrix(a, b, max_d2=5)
    np.testing.assert_allclose(
        C,
        [
            [5, 1],
            [np.nan, 2],
            [np.nan, 5]
        ]
    )    

def test_norm2squared_empty():
    a = []
    b = np.array([[0., 0],[1., 1]])
    C = mm.distances.norm2squared_matrix(a, b)
    assert C.size == 0
    C = mm.distances.norm2squared_matrix(b, a)
    assert C.size == 0

def test_iou_matrix():
    a = np.array([
        [0, 0, 1, 2],
    ])

    b = np.array([
        [0, 0, 1, 2],
        [0, 0, 1, 1],
        [1, 1, 1, 1],
        [0.5, 0, 1, 1],
        [0, 1, 1, 1],
    ])
    np.testing.assert_allclose(
        mm.distances.iou_matrix(a, b),
        [[0, 0.5, 1, 0.8, 0.5]],
        atol=1e-4
    )

    np.testing.assert_allclose(
        mm.distances.iou_matrix(a, b, max_iou=0.5),
        [[0, 0.5, np.nan, np.nan, 0.5]],
        atol=1e-4
    )

