# -*- coding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from functools import partial

import megengine.module as M
from megengine.module.normalization import GroupNorm, InstanceNorm, LayerNorm


def get_norm(norm, channels=None, **kwargs):
    """
    Args:
        norm (str): currently support "BN", "SyncBN", "FrozenBN", "GN", "LN" and "IN"
        channels (int): Norm channels, default
        kwargs (dict): extra params like affine, trac

    Returns:
        M.Module or None: the normalization layer
    """
    if norm is None:
        return None
    norm = {
        "BN": M.BatchNorm2d,
        "SyncBN": M.SyncBatchNorm,
        "FrozenBN": partial(M.BatchNorm2d, freeze=True),
        "GN": GroupNorm,
        "LN": LayerNorm,
        "IN": InstanceNorm,
    }[norm]
    if channels is not None:
        if norm.__name__ == "GroupNorm":
            num_groups = kwargs.pop("num_groups", 32)
            return norm(num_groups, channels, **kwargs)
        else:
            return norm(channels, **kwargs)
    return norm
