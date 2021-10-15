#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import contextlib

import megengine.module as M
from megengine import Tensor

NORM = (
    M.BatchNorm1d,
    M.BatchNorm2d,
    M.SyncBatchNorm,
)


def freeze(module):
    """
    func used to freeze module, detach parameters recurrsively

    NOTE:
        this func might not work.
    """
    named_children = list(module.named_children())
    if len(named_children) == 0:
        # TODO wangfeng02 check if it works for GN/LN/IN
        if isinstance(module, NORM):
            # convert norm to frozen bn
            module.freeze = True
        else:
            for k, v in vars(module).items():
                if isinstance(v, Tensor):
                    weights = getattr(module, k).detach()
                    setattr(module, k, weights)

    for name, children in named_children:
        freeze_module = freeze(children)
        setattr(module, name, freeze_module)
    return module


@contextlib.contextmanager
def adjust_stats(module, training=False):
    """Adjust module to training/eval mode temporarily.

    Args:
        module (M.Module): used module.
        training (bool): training mode. True for train mode, False fro eval mode.
    """
    backup_stats = {}

    def recursive_backup_stats(module, mode):
        for m in module.modules():
            # save prev status to dict
            backup_stats[m] = m.training
            m.train(mode, recursive=False)

    def recursive_recover_stats(module):
        for m in module.modules():
            # recover prev status from dict
            m.training = backup_stats.pop(m)

    recursive_backup_stats(module, mode=training)
    yield module
    recursive_recover_stats(module)
