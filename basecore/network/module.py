#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# This file includes functions of module level operator

import contextlib

import megengine as mge
import megengine.functional as F
import megengine.module as M

__all__ = [
    "adjust_stats",
    "freeze",
    "freeze_norm",
    "fuse_conv_and_bn",
    "replace_module",
    "_NORM",
]

_NORM = (
    M.BatchNorm1d,
    M.BatchNorm2d,
    M.SyncBatchNorm,
)


@contextlib.contextmanager
def adjust_stats(module: M.Module, training=False):
    """Adjust module to training/eval mode temporarily.

    Args:
        module: used module.
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


def freeze(module):
    """
    func used to freeze module, detach parameters recurrsively

    NOTE:
        this func might not work.
    """
    named_children = list(module.named_children())
    if len(named_children) == 0:
        # TODO wangfeng02 check if it works for GN/LN/IN
        if isinstance(module, _NORM):
            # convert norm to frozen bn
            module.freeze = True
        else:
            for k, v in vars(module).items():
                if isinstance(v, mge.Tensor):
                    weights = getattr(module, k).detach()
                    setattr(module, k, weights)

    for name, children in named_children:
        freeze_module = freeze(children)
        setattr(module, name, freeze_module)
    return module


def freeze_norm(module: M.Module) -> M.Module:
    """
    Freeze all batchnorm module in a given module.

    Args:
        module: module to freeze.
    """
    for m in module.modules():
        if isinstance(m, _NORM):
            m.freeze = True
    return module


def fuse_conv_and_bn(conv: M.Conv2d, bn: M.BatchNorm2d) -> M.Conv2d:
    """
    Fuse megengine convolution and batchnorm layers into a single conv layer.
    NOTE that now we only support 2d Conv and BN.

    Args:
        conv: convolution layer.
        bn: batchnorm layer.
    """
    fused_conv = M.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True,
    )

    # fused_conv.weight = bn.weight / running_var * conv.weight
    w_conv = conv.weight.reshape(conv.out_channels, -1)
    factor = (bn.weight / F.sqrt(bn.eps + bn.running_var)).reshape(-1)
    fused_conv.weight = mge.Parameter(
        (factor.reshape(-1, 1) * w_conv).reshape(fused_conv.weight.shape)
    )

    # fused_conv.bias = bn.bias + (conv.bias - running_mean) * bn.weight / runing_var
    conv_bias = F.zeros(bn.running_mean.shape) if conv.bias is None else conv.bias
    fuse_bias = bn.bias + (conv_bias - bn.running_mean) * factor.reshape(1, -1, 1, 1)
    fused_conv.bias = mge.Parameter(fuse_bias)

    return fused_conv


def replace_module(
    module: M.Module, replaced_module_type, new_module_type, replace_func=None
) -> M.Module:
    """
    Replace given type in module to a new type. mostly used in deploy.

    Args:
        module : model to apply replace operation.
        replaced_module_type (Type): module type to be replaced.
        new_module_type (Type)
        replace_func (function): python function to describe replace logic. Defalut value None.

    Returns:
        model: module that already been replaced.
    """
    def default_replace_func(replaced_module_type, new_module_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recurrsively replace
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type, new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model
