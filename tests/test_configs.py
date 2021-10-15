#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import pickle
import unittest
from copy import deepcopy

from basecore.config import ConfigDict


class TestConfig(unittest.TestCase):

    def setUp(self):
        self.dict1 = {"MODEL": {"NAME": "RetinaNet", "FPN": True}}
        self.dict2 = {"MODEL": {"NAME": "FCOS"}}
        self.list1 = ["A.B.C", (1, 2), "A.B.D", 3]
        self.list2 = ["A.B.C", {"A": 1, "B": 2}, "A.B.D", 3]

    def tearDown(self):
        pass

    def test_generate_object(self):
        try:
            cfg = ConfigDict(self.dict1)
            cfg2 = ConfigDict(cfg)
            cfg2 = ConfigDict(self.dict1, TEST=True)
        except Exception:
            self.fail("raise Exception unexceptedly!")
        self.assertTrue(cfg2.TEST)

    def test_eq_func(self):
        cfg = ConfigDict(self.dict1)
        self.assertFalse(cfg == 1)

        cfg2 = deepcopy(cfg)
        self.assertTrue(cfg == cfg2)

        # modify key
        cfg2.MODEL.NAME = "FCOS"
        self.assertFalse(cfg == cfg2)

        # add key
        cfg2 = deepcopy(cfg)
        cfg2.TEST = True
        self.assertFalse(cfg == cfg2)

    def test_diff_func(self):
        cfg1 = ConfigDict(self.dict1)
        cfg2 = ConfigDict(self.dict2)

        diff1 = cfg1.diff(cfg2)
        diff2 = cfg2.diff(cfg1)
        self.assertNotEqual(diff1, diff2)

    def test_update_func(self):
        cfg1 = ConfigDict(self.dict1)
        cfg1.update(self.dict2)
        self.assertEqual(cfg1, ConfigDict(["MODEL.NAME", "FCOS"]))

    def test_merge_func(self):
        cfg1 = ConfigDict(self.dict1)
        cfg1.merge(self.dict2)
        cfg1.merge(self.list1)
        updated_cfg = {"MODEL": {"FPN": True, "NAME": "FCOS"}, "A": {"B": {"C": (1, 2), "D": 3}}}
        self.assertEqual(cfg1, ConfigDict(updated_cfg))

    def test_merge_with_literal_eval(self):
        cfg1 = ConfigDict(self.dict1)
        cfg1.merge(["MODEL.FPN", "True"])
        self.assertTrue(cfg1.MODEL.FPN)

    def test_find_func(self):
        cfg1 = ConfigDict(self.dict1)
        cfg1.merge(self.list1)
        find_res = cfg1.find("B", show=False)
        self.assertEqual(find_res, ConfigDict(self.list1))

    def test_uion_func(self):
        cfg1 = ConfigDict(self.dict1)
        cfg1.merge(self.list1)
        cfg2 = ConfigDict(self.dict1)
        cfg2.MODEL.FPN = False
        un = cfg1.union(cfg2)
        self.assertEqual(un, ConfigDict(["MODEL.NAME", "RetinaNet"]))

    def test_remove_func(self):
        cfg1 = ConfigDict(self.dict1)
        cfg1.merge(self.list1)
        cfg2 = ConfigDict(self.dict1)
        cfg1.remove(cfg2)
        self.assertEqual(cfg1, ConfigDict(self.list1))

    def test_pickle(self):
        cfg1 = ConfigDict(self.dict1)
        filename = "temp.pkl"
        with open(filename, "wb") as f:
            pickle.dump(cfg1, f)

        with open(filename, "rb") as f:
            cfg2 = pickle.load(f)
        os.remove(filename)
        x = cfg1.diff(cfg2)
        assert len(x.keys()) == 0

    def test_hash(self):
        cfg1 = ConfigDict(self.list1)
        cfg2 = ConfigDict(self.list1)
        cfg3 = ConfigDict(self.list2)
        assert hash(cfg1) == hash(cfg2)
        assert hash(cfg1) != hash(cfg3)

    def test_yaml_load_and_save(self):
        x = ConfigDict(values_or_file="test.yaml")
        temp_file = "temp.yaml"
        x.dump_to_file(temp_file)
        y = ConfigDict(temp_file)
        assert len(x.diff(y).keys()) == 0
        os.remove(temp_file)

    def test_bool_repr(self):
        cfg1 = ConfigDict(dict(solver=dict(ema=dict(enabled=True))))
        repr(cfg1)
        cfg2 = ConfigDict(dict(solver=dict(ema=dict(enabled=True, momentum=0.999))))
        repr(cfg2)
        cfg3 = ConfigDict(dict(solver=dict(ema=dict(enabled=True, momentum=0.999, alpha=None))))
        repr(cfg3)
        cfg4 = ConfigDict(dict(ema=dict(enabled=True, momentum=0.999)))
        repr(cfg4)
