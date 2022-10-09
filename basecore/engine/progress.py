#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

from collections import OrderedDict
from typing import List
from easydict import EasyDict


class Progress(EasyDict):
    """
    Progress class is used to record the progress of training.
    Default progress key is ["epoch", "iter"].

    NOTE: The progress starts from 1. For example, for a 5 iter training,
        the iter info is [1, 2, 3, 4, 5].

    Example:
        >>> p = Progress()
        >>> while not p.reach_epoch_end():
        >>>     while not p.reach_iter_end():
        >>>         # doing something
        >>>         p.next_iter()
        >>>     p.next_epoch()

    Args:
        progress_key (list): the key of progress, default is ["epoch", "iter"].
        max_info (list): the max value of each progress key, deafult to [100, 1000].
    """

    def __init__(self, progress_key=None, max_info=None):
        if progress_key is None:
            self._key = ["epoch", "iter"]
        else:
            self._key = list(progress_key)

        if max_info is None:
            max_info = [100, 1000]

        if len(self._key) != len(max_info):
            raise ValueError("The length of progress_key and max_info should be the same.")

        for k, max_val in zip(self._key, max_info):
            setattr(self, k, 1)
            setattr(self, "max_" + k, max_val)

    def progress_str_list(self, connect_str: str = "") -> List[str]:
        return [
            "{}{}{}".format(k, connect_str, getattr(self, k, None))
            for k in self._key
        ]

    def scale_to_iterwise(self, epoch_info_list) -> List[int]:
        return [x * self.max_iter for x in epoch_info_list]

    def scale_to_epochwise(self, iter_info_list) -> List[int]:
        return [x // self.max_iter for x in iter_info_list]

    def current_iter(self) -> int:
        return (self.epoch - 1) * self.max_iter + self.iter - 1

    def left_iter(self) -> int:
        return self.max_epoch * self.max_iter - self.current_iter()

    def is_first_iter(self) -> bool:
        return self.epoch == 1 and self.iter == 1

    def state_dict(self) -> OrderedDict:
        return OrderedDict({k: getattr(self, k, None) for k in self._key})

    def load_state_dict(self, loaded_dict):
        self._key = list(loaded_dict.keys())
        self.update(loaded_dict)

    def reach_epoch_end(self) -> bool:
        return self.epoch > self.max_epoch

    def reach_iter_end(self) -> bool:
        return self.iter > self.max_iter

    def next_iter(self):
        self.iter += 1

    def next_epoch(self):
        self.epoch += 1
        self.iter = 1

    def __str__(self) -> str:
        progress_str = list()
        for k in self._key:
            cur_value = getattr(self, k, None)
            max_value = getattr(self, "max_" + k, None)
            p_str = "{}:{}/{}".format(k, cur_value, max_value)
            progress_str.append(p_str)
        return ", ".join(progress_str)
