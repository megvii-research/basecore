#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import megengine.module as M


def get_activation(name, **kwargs):
    if not name:
        return None

    activation = {
        "identity": M.Identity,  # identity is different from None
        "relu": M.ReLU,
        "prelu": M.PReLU,
        "sigmoid": M.Sigmoid,
    }
    act = activation[name](**kwargs)
    return act
