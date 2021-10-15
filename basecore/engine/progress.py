#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from collections import OrderedDict
from easydict import EasyDict


class Progress(EasyDict):

    def __init__(self, progress_key=None):
        if progress_key is None:
            self._key = ["epoch", "iter"]
        else:
            self._key = list(progress_key)

    def __str__(self):
        progress_str = list()
        for k in self._key:
            cur_value = getattr(self, k, None)
            max_value = getattr(self, "max_" + k, None)
            p_str = "{}:{}/{}".format(k, cur_value, max_value)
            progress_str.append(p_str)
        return ", ".join(progress_str)

    def progress_str_list(self, connect_str=""):
        return [
            "{}{}{}".format(k, connect_str, getattr(self, k, None))
            for k in self._key
        ]

    def state_dict(self):
        return OrderedDict({k: getattr(self, k, None) for k in self._key})

    def load_state_dict(self, loaded_dict):
        self._key = list(loaded_dict.keys())
        self.update(loaded_dict)
