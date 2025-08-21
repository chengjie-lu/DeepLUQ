#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/29 下午3:12
# @Author  :
# @File    : test_metrics.py
# @Software: PyCharm

import numpy as np
import pytest
from src.deepluq.metrics_dl import DLMetrics  # <-- adjust import if your class is in another file


@pytest.fixture
def uq():
    """Fixture to provide a fresh UQMetrics instance for each test."""
    return DLMetrics()


# -------------------------------
# Classification uncertainty metrics
# -------------------------------
def test_cal_vr(uq):
    events = [[0.2, 0.8], [0.1, 0.9], [0.6, 0.4]]
    vr = uq.cal_vr(events)
    assert 0 <= vr <= 1
    assert isinstance(vr, float)


def test_calcu_entropy(uq):
    events = [0.5, 0.5]
    entropy = uq.calcu_entropy(events)
    assert pytest.approx(entropy, 0.01) == 1.0  # entropy of uniform distribution (base 2)


def test_calcu_mi(uq):
    events = np.array([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]])
    mi = uq.calcu_mi(events)
    assert isinstance(mi, float)


# -------------------------------
# Total variance metrics
# -------------------------------
def test_calcu_tv_center_point(uq):
    matrix = np.array([[1, 2], [3, 4], [5, 6]])
    val = uq.calcu_tv(matrix, "center_point")
    assert val > 0


def test_calcu_tv_bounding_box(uq):
    matrix = np.array([[1, 2], [3, 4], [5, 6]])
    val = uq.calcu_tv(matrix, "bounding_box")
    assert val > 0


def test_calcu_tv_invalid_tag(uq):
    matrix = np.array([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError):
        uq.calcu_tv(matrix, "invalid")


# -------------------------------
# Mutual Information
# -------------------------------
def test_calcu_mutual_information(uq):
    X = np.array([0, 0, 1, 1])
    Y = np.array([0, 1, 0, 1])
    Z = np.array([1, 1, 0, 0])
    mi = uq.calcu_mutual_information(X, Y, Z)
    assert isinstance(mi, float)
    assert mi >= 0


# -------------------------------
# Geometric uncertainty metrics
# -------------------------------
def test_calcu_prediction_surface(uq):
    # Four boxes (enough to form convex hulls)
    boxes = [
        [0, 0, 1, 1],
        [1, 0, 2, 1],
        [0, 1, 1, 2],
        [1, 1, 2, 2],
    ]
    surface = uq.calcu_prediction_surface(boxes)
    assert surface >= 0


def test_calcu_prediction_surface_too_few_points(uq):
    boxes = [[0, 0, 1, 1]]  # not enough points for hull
    surface = uq.calcu_prediction_surface(boxes)
    assert surface == -1
