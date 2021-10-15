#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from typing import Tuple, Union

import megengine.module as M

from .activation import get_activation
from .module import freeze
from .norm import get_norm


# TODO change to another way
class ConvNormActivation2d(M.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        conv_mode: str = "CROSS_CORRELATION",
        compute_mode: str = "DEFAULT",
        norm_name="BN",
        eps=1e-5,
        momentum=0.9,
        affine=True,
        track_running_stats=True,
        freeze_conv=False,
        freeze_norm=False,
        activation_name="relu",
        **kwargs
    ):
        super().__init__()
        conv = M.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            conv_mode,
            compute_mode,
        )
        if freeze_conv:
            conv = freeze(conv)
        self.conv = conv

        norm = get_norm(norm_name)(out_channels, eps, momentum, affine, track_running_stats)
        if freeze_norm:
            norm = freeze(norm)
        self.norm = norm

        self.act = get_activation(activation_name, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class SandBox:
    """
    A wrapper to change given pyobj to another type but keep all methods.
    Such an wrapper could make megengine module untraceable.
    """

    def __init__(self, pyobj):
        # copy data
        self.__dict__ = pyobj.__dict__

        py_class = pyobj.__class__
        skip_attr = (
            # self defined attr
            "__init__", "__class__", "__repr__", "__dict__",
            # get/set related attr
            "__getattribute__", "__setattr__",
        )
        for attr in dir(py_class):
            if attr in skip_attr:
                continue
            attr_val = getattr(py_class, attr)
            setattr(self.__class__, attr, attr_val)
        self.__obj = pyobj

    def __repr__(self):
        module_str = self.__obj.__repr__()
        indent_str = [self.__class__.__name__]
        indent_str.extend([" " * 2 + s for s in module_str.split("\n")])
        indent_str.append(")")
        return "\n".join(indent_str)
