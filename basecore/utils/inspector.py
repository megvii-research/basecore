#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import inspect
import os
import time

_CALL_TIME_RECORDER = {}
_CALL_COUNT_RECORDER = {}

__all__ = [
    "get_call_count",
    "get_caller_basedir",
    "get_caller_context",
    "get_last_call_deltatime",
]


def get_caller_context(depth=0):
    """
    Args:
        depth (int): Depth of caller conext, use 0 for caller depth. Default value: 0.

    Returns:
        str: module name of the caller
        tuple: a hashable key to be used to identify different callers
    """
    # following logic is a little bit faster than inspect.stack() logic
    frame = inspect.currentframe().f_back
    for _ in range(depth):
        frame = frame.f_back

    while frame:
        code = frame.f_code
        if os.path.join("utils", "logging.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back


def get_caller_basedir():
    """get the file where the caller locates"""
    frames = inspect.getouterframes(inspect.currentframe())
    return os.path.dirname(os.path.realpath(frames[1][1]))


def get_last_call_deltatime(reset=True, reset_seconds=1, depth=0):
    """get delta time between last two calls.

    Args:
        reset (bool): If reset is False, last calltime will be updated only if
            delta time is greater than reset seconds. Default value: True.
        reset_seconds (int): Value of reset seconds. This argument only works
            if reset is False. Default value: 1.
        depth (int): Depth of caller conext, use 0 for caller depth. Default value: 0.
    """
    caller_module, key = get_caller_context(depth=depth + 1)
    last_calltime = _CALL_TIME_RECORDER.get(key, None)
    current_time = time.time()
    if last_calltime is not None:
        delta_time = current_time - last_calltime
        if reset is True or delta_time >= reset_seconds:
            _CALL_TIME_RECORDER[key] = current_time
        return delta_time
    else:
        _CALL_TIME_RECORDER[key] = current_time


def get_call_count(depth=0):
    """get call count, starts from 1.

    Args:
        depth (int): Depth of caller conext, use 0 for caller depth. Default value: 0.
    """
    caller_module, key = get_caller_context(depth=depth + 1)
    call_count = _CALL_COUNT_RECORDER.get(key, 0) + 1
    _CALL_COUNT_RECORDER[key] = call_count
    return call_count
