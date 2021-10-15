#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import unittest

import megengine as mge
import megengine.module as M

from basecore.network import adjust_stats


class TestLayers(unittest.TestCase):

    def test_adjust_stats(self):
        data = mge.functional.ones((1, 10, 800, 800))
        # use bn since bn changes state during train/val
        model = M.BatchNorm2d(10)
        prev_state = model.state_dict()
        with adjust_stats(model, training=False) as model:
            model(data)
        self.assertTrue(len(model.state_dict()) == len(prev_state))
        for k, v in prev_state.items():
            self.assertTrue(all(v == model.state_dict()[k]))

        # test under train mode
        prev_state = model.state_dict()
        with adjust_stats(model, training=True) as model:
            model(data)
        self.assertTrue(len(model.state_dict()) == len(prev_state))
        equal_res = [all(v == model.state_dict()[k]) for k, v in prev_state.items()]
        self.assertFalse(all(equal_res))

        # test recurrsive case
        prev_state = model.state_dict()
        with adjust_stats(model, training=False) as model:
            with adjust_stats(model, training=False) as model:
                model(data)
        self.assertTrue(len(model.state_dict()) == len(prev_state))
        for k, v in prev_state.items():
            self.assertTrue(all(v == model.state_dict()[k]))
