#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/29 下午3:12
# @Author  :
# @File    : test_metrics.py
# @Software: PyCharm

import pytest
from src import deepluq
import unittest


# uq_metrics = UQMetrics()
# uq_metrics.calcu_entropy([1 / 3, 1 / 3, 1 / 3])
# print(uq_metrics.shannon_entropy)
#
# print(uq_metrics.cal_vr(events=[[1 / 3, 1 / 3, 1 / 3]]))
# bs = [
#     [
#         1013.3162231445312,
#         1310.352294921875,
#         1118.556884765625,
#         1385.857177734375
#     ],
#     [
#         1014.5834350585938,
#         1308.5045166015625,
#         1121.2974853515625,
#         1388.34228515625
#     ],
#     [
#         1015.1859130859375,
#         1308.117431640625,
#         1119.5179443359375,
#         1386.121826171875
#     ]
# ]
#
# uq_metrics.calcu_prediction_surface(bs)


class TestMetrics(unittest.TestCase):
    # def __init__(self):
    #     super().__init__()
    #     self.uq_metrics = deepluq.metrics.UQMetrics()

    def test_calcu_entropy(self):
        self.uq_metrics = deepluq.metrics
        self.uq_metrics.calcu_entropy([1 / 3, 1 / 3, 1 / 3])
        print(self.uq_metrics.shannon_entropy)
