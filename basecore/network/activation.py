#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import megengine.module as M


__all__ = ["get_activation", "register_activation"]


_ACTIVATION = {
    "identity": M.Identity,  # identity is different from None
    "relu": M.ReLU,
    "prelu": M.PReLU,
    "sigmoid": M.Sigmoid,
    "silu": M.SiLU,
}


def get_activation(name, **kwargs):
    if not name:
        return None

    act = _ACTIVATION[name](**kwargs)
    return act


def register_activation(name, module):
    assert name not in _ACTIVATION
    _ACTIVATION[name] = module
