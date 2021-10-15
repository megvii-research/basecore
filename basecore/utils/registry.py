#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import pprint
from typing import Dict, Optional
from tabulate import tabulate


# design of Registry is inspired by fvcore, please check
# https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py
# for more details
class Registry:
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        OPTIMIZER = Registry('optimizer')

    To register an object:

    .. code-block:: python

        @OPTIMIZER.register()
        class MyOptimizer():
            ...

    Or:

    .. code-block:: python

        OPTIMIZER.register(MyOptimizer)

    Or:

    .. code-block:: python

        @OPTIMIZER.register("Name for Registry")
        class MyOptimizer():
            ...
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, object] = {}

    def _do_register(self, name: str, obj: object) -> None:
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(self, obj: object = None, name: str = None) -> Optional[object]:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: object) -> object:
                nonlocal name
                if name is None:
                    name = func_or_class.__name__  # pyre-ignore
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        if name is None:
            name = obj.__name__  # pyre-ignore
        self._do_register(name, obj)

    def get(self, name: str) -> object:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(name, self._name)
            )
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        table_content = [(k, pprint.pformat(v)) for k, v in self._obj_map.items()]
        table = tabulate(table_content, headers=table_headers, tablefmt="fancy_grid")
        return "Registry of {}:\n".format(self._name) + table

    def items(self):
        return self._obj_map.items()

    def keys(self):
        return self._obj_map.keys()

    def values(self):
        return self._obj_map.values()
