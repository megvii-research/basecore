#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import copy
import itertools
import pprint
import re
from ast import literal_eval
from typing import Collection, Iterable, Mapping, Sequence
import yaml
from colorama import Back, Fore, Style
from tabulate import tabulate

import numpy as np


# from https://stackoverflow.com/questions/32954486
def zip_equal(*iterables):
    """if length of iterators are not equal, an exception will be raised"""
    sentinel = object()
    for combo in itertools.zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError('Iterables have different lengths')
        yield combo


def iterable_to_dict(iterator):
    """
    Convert an iterable object to dict.  Note that values in even index will be casted
    using `ast.literal_eval`.  Value like "False" will become bool False.
    Try "'False'" if a string "False" is needed.
    """
    assert isinstance(iterator, Iterable), "input must be iterable"
    try:
        values_dict = {
            k: v for k, v in zip_equal(
                itertools.islice(iterator, 0, None, 2),
                itertools.islice(iterator, 1, None, 2)
            )
        }
    except ValueError:
        raise ValueError("length of iterator is not even number.")

    # values from CLI are always treated as string type, try to cast value using literal_eval
    for k, v in values_dict.items():
        try:
            v = literal_eval(v)
        except Exception:
            pass
        values_dict[k] = v

    ret_dict = {}
    for k, v in values_dict.items():
        split_keys = k.split(".")
        cur_dict = ret_dict
        for split_k in split_keys[:-1]:
            if split_k not in cur_dict.keys():
                cur_dict[split_k] = dict()
            cur_dict = cur_dict[split_k]
        cur_dict[split_keys[-1]] = v
    return ret_dict


def highlight(keyword, target, flags=0, color=Fore.BLACK + Back.YELLOW):
    """
    use given color to highlight keyword in target string

    Args:
        keyword (str): highlight string
        target (str): target string
        color (str): string represent the color, use black foreground
            and yellow background as default
    """
    return re.sub(keyword, color + r"\g<0>" + Style.RESET_ALL, target, flags=flags)


def is_equal(x, y):
    """judge if x is equal to y from inner values"""
    if type(x) != type(y):
        return False
    if isinstance(x, np.ndarray):
        return np.array_equal(x, y)
    val_eq = (x == y)
    if isinstance(val_eq, Collection):
        val_eq = all(val_eq)
    return val_eq


def file_to_dict(filename):
    """convert a yaml file to dict"""
    def recurrive_eval(d):
        for k, v in d.items():
            if isinstance(v, Mapping):
                d[k] = recurrive_eval(v)
            else:
                try:
                    eval_v = literal_eval(v)
                    d[k] = eval_v
                except Exception:
                    pass
        return d

    with open(filename, "r") as f:
        safe_dict = yaml.load(f, Loader=yaml.Loader)
    return recurrive_eval(safe_dict)


class ConfigDict(dict):
    """
    Basic configuration for all models, which provides useful logic for
        inner MEGVII platform.
    Note that params in kwargs should be the latest value.
    """
    # TODO: support check mode
    MODE = None

    def __init__(self, values_or_file=None, **kwargs):
        """
        Args:
            values_or_file ():
            **kwargs
        """
        self.merge(values_or_file, **kwargs)

    def set_mode(self, mode=None):
        """set config mode

        Args:
            mode (str):
        """
        if mode is not None:
            assert mode in ["freeze"]
        self.MODE = mode
        for v in self.values():
            if isinstance(v, ConfigDict):
                v.set_mode(mode)

    def _check_mode(self):
        if self.MODE == "freeze":
            raise ValueError("Delete value under 'freeze' mode! Change mode to delete attr")

    def merge(self, values_or_file, **kwargs):
        """
        merge all key and values of config as ConfigDict's attributes.
        Note: kwargs will override values in config if they have the same keys.
        If using iterable argument, values in even index will be casted using literal_eval.
        Value like "False" will become bool False. Try "'False'" if a string "False" is needed.

        Args:
            values_or_file (Union[list, str, tuple, dict]): could be a list/tuple
                which len() % 2 == 0, a dict or a filename describe in string.
        """
        def recursive_merge(src, dst):
            """resursive merge from a dict to target dict

            Args:
                src (dict): target dict, this dict will be updated after merge
                dst (dict): dict where updated value comes from.
            """
            for dst_key, dst_value in dst.items():
                src_value = src.get(dst_key, ConfigDict())
                if isinstance(dst_value, Mapping) and isinstance(src_value, Mapping):
                    setattr(src, dst_key, recursive_merge(src_value, dst_value))
                else:
                    setattr(src, dst_key, dst_value)
            return src

        self._check_mode()
        if values_or_file is None:
            values_or_file = {}
        if isinstance(values_or_file, str):  # filename
            values_or_file = file_to_dict(values_or_file)
        elif isinstance(values_or_file, Sequence):
            values_or_file = iterable_to_dict(values_or_file)

        assert isinstance(values_or_file, Mapping), \
            "unsupported value type: {}".format(type(values_or_file))

        recursive_merge(self, values_or_file)
        if kwargs:
            recursive_merge(self, kwargs)
        return self

    def update(self, values, **kwargs):
        """
        values: could be a list/tuple which len() % 2 == 0
            could be a dict or a config
        """
        self._check_mode()
        if isinstance(values, Sequence) and not isinstance(values, str):
            values = iterable_to_dict(values)
        values.update(**kwargs)
        for k, v in values.items():
            self.__setitem__(k, v)

    def pop(self, k, d=None):
        self._check_mode()
        try:
            super().__delattr__(k)
        except Exception:
            pass
        return super().pop(k, d)

    def diff(self, cfg):
        """equal to use self - (self & cfg) but faster

        cfg(ConfigDict): pass
        """
        def recursive_diff(src, dst):
            """find difference between src dict and dst dict"""
            diff_result = {}
            for k, v in src.items():
                if k not in dst:
                    diff_result[k] = v
                elif not is_equal(v, dst[k]):
                    if isinstance(v, Mapping):
                        diff_result[k] = recursive_diff(v, dst[k])
                    else:
                        diff_result[k] = v
            return diff_result

        if not isinstance(cfg, ConfigDict):
            cfg = ConfigDict(cfg)
        diff_result = recursive_diff(self, cfg)
        return ConfigDict(diff_result)

    def find(self, key: str, case_sense=False, show=True, color=Fore.BLACK + Back.YELLOW):
        """
        find a given key and its value in config

        Args:
            key (str): the string you want to find
            show (bool): if show is True, print find result; or return the find result.
            case_sense (bool): if True, use case sensetive search.
            color (str): color of `key`, default color is black(foreground) yellow(background).
        """

        def recursive_find(param_dict: dict, key: str) -> dict:
            find_result = {}
            for k, v in param_dict.items():
                if re.search(key, k, flags):
                    find_result[k] = v
                elif isinstance(v, Mapping):
                    res = recursive_find(v, key)
                    if res:
                        find_result[k] = res
            return find_result

        flags = 0 if case_sense else re.IGNORECASE

        find_result = recursive_find(self, key)
        find_result = ConfigDict(find_result)
        if show:
            print(highlight(key, repr(find_result), flags, color))
            return
        return find_result

    def union(self, cfg):
        """get union of current config and provided config.

        Args:
            cfg (ConfigDict): provided config. If not ConfigDict, it will be autocasted.
        """
        def recursive_union(src, dst):
            union_result = {}
            for k, v in src.items():
                if k in dst:
                    dst_v = dst.get(k)
                    if v == dst_v:
                        union_result[k] = v
                    elif isinstance(v, Mapping) and isinstance(dst_v, Mapping):
                        ret = recursive_union(v, dst_v)
                        if ret:
                            union_result[k] = ret
            return union_result

        if not isinstance(cfg, ConfigDict):
            cfg = ConfigDict(cfg)
        return ConfigDict(recursive_union(self, cfg))

    def remove(self, cfg, match_value=True):
        """remove keys and values provided from cfg.

        Args:
            cfg (ConfigDict): provided config. If cfg is not ConfigDict, it will be autocasted.
            match_value (bool): if False, removed values have no need to be the same.
                if True, remove form current config only if keys and values both match.
        """
        def recursive_remove(src, dst):
            for k, v in dst.items():
                if k in src:
                    src_v = src.get(k)
                    if isinstance(src_v, Mapping):
                        if isinstance(v, Mapping):
                            recursive_remove(src_v, v)
                        else:
                            src.pop(k)
                    else:
                        # key reaches the end
                        src.pop(k)
            poped_keys = [k for k, v in src.items() if not v]
            for k in poped_keys:
                src.pop(k)

        if not isinstance(cfg, ConfigDict):
            cfg = ConfigDict(cfg)
        if match_value:
            cfg = self.union(cfg)
        recursive_remove(self, cfg)

    def clone(self):
        return copy.deepcopy(self)

    @property
    def depth(self):
        """
        Get depth of ConfigDict where `depth` is defined as dict level.
        """
        def recursive_depth(obj):
            if isinstance(obj, ConfigDict):
                return max([recursive_depth(v) for v in obj.values()]) + 1
            else:  # root case
                return 0

        return recursive_depth(self)

    def show_unused_attr(self):
        # TODO wangfeng02: refine in the future.
        unused_list = [
            k for k in vars(self).keys() if not k.startswith("_") and k not in self._used_attr
        ]
        return unused_list

    def to_dict(self):
        ret_dict = {}
        for k, v in self.items():
            if isinstance(v, ConfigDict):
                v = v.to_dict()
            ret_dict[k] = v
        return ret_dict

    def dump_to_file(self, filename, **kwargs):
        with open(filename, "w") as f:
            dump_dict = self.to_dict()
            yaml.dump(dump_dict, stream=f, **kwargs)

    def _convert(self, value):
        if isinstance(value, (list, tuple)):
            value = type(value)(ConfigDict(x) if isinstance(x, dict) else x for x in value)
        elif isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        return value

    def __setattr__(self, name, value):
        value = self._convert(value)
        super().__setattr__(name, value)
        if not hasattr(ConfigDict, name):
            # make sure class level variable not shown in config
            super().__setitem__(name, value)

    def __setitem__(self, key, value):
        value = self._convert(value)
        super().__setattr__(key, value)
        super().__setitem__(key, value)

    def __delattr__(self, name):
        self.pop(name)

    def __eq__(self, cfg):
        def recursive_eq(src, dst):
            if len(src.keys()) != len(dst.keys()):
                return False
            for k, v in src.items():
                if k not in dst:
                    return False
                dst_v = dst.get(k)
                if type(v) != type(dst_v):
                    return False
                if isinstance(v, Mapping):
                    # dict type
                    if not recursive_eq(v, dst_v):
                        return False
                else:
                    # non dict type
                    if not is_equal(v, dst_v):
                        return False
            return True

        if type(self) == type(cfg):
            return recursive_eq(self, cfg)
        return False

    def __hash__(self):
        sorted_items = sorted(self.to_dict().items())
        return hash(str(sorted_items))

    def __add__(self, cfg):
        ret_cfg = self.clone()
        ret_cfg.merge(cfg)
        return ret_cfg

    def __iadd__(self, cfg):
        self.merge(cfg)
        return self

    def __sub__(self, cfg):
        ret_cfg = self.clone()
        ret_cfg.remove(cfg)
        return ret_cfg

    def __isub__(self, cfg):
        self.remove(cfg)
        return self

    def __and__(self, cfg):
        return self.union(cfg)

    def __repr__(self):
        config_table = []
        table_header = ["keys", "values"]
        for k, v in self.items():
            if k.startswith("_"):
                continue
            # force cast bool to int in displaying to work around the following issue:
            # https://github.com/astanin/python-tabulate/issues/135
            # note that tabulate will display in the right way.
            if isinstance(v, bool):
                v = int(v)
            config_table.append((str(k), pprint.pformat(v),))

        return tabulate(config_table, headers=table_header, tablefmt="fancy_grid")
