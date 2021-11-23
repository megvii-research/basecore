#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import contextlib
from typing import Callable, Mapping, Sequence

import megengine as mge
from megengine.autodiff import GradManager
from megengine.optimizer import Optimizer

try:
    import megengine.amp as amp
    from megegnine.amp import GradScaler
except ImportError:
    GradScaler = None

__all__ = ["Solver"]


@contextlib.contextmanager
def nullcontext():
    yield


class Solver:

    """
    Attributes:
        optimizer (Optimizer): megengine Optimizer.
        grad_manager (GradManager): megengine GradManager.
        grad_scaler (GradScaler): megengine GradScaler. Defaults to None.
            If scaler is not None, apply mixed-precision training.
        grad_clip_fn (Callable): function that describes gradient clipping process.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        grad_manager: GradManager,
        grad_scaler: GradScaler = None,
        grad_clip_fn: Callable = None,
    ):
        self.optimizer = optimizer
        self.grad_manager = grad_manager
        self.grad_scaler = grad_scaler
        self.enable_amp = self.grad_scaler is not None
        self.grad_clip_fn = grad_clip_fn

    def minimize(self, func, *args, step_grad=True, **kwargs):
        """
        Auto backward losses and step gradient. losses is provided by function.
        Backward will be applied on key named "total_loss" if dict is returned, otherwise,
        all keys with "loss" in its name will be applied.

        Args:
            func (callable): function which describes to minimize.
            args (tuple): function args.
            step_grad (bool): wheather step and clear gradient or not. Default to `True`.
            kwargs (dict): function keyword args.

        Returns:
            return whatever function returns.

        Examples:

        .. code-block::

            # common usage
            solver.minimize(model, model_input)

            # step grad every 4 iters
            solver.minimize(model, model_input, step_grad=iter % 4==0)
        """
        amp_context = amp.autocast if self.enable_amp else nullcontext
        with self.grad_manager:
            with amp_context():
                outputs = func(*args, **kwargs)

                if isinstance(outputs, mge.Tensor):
                    total_loss = outputs
                elif isinstance(outputs, Mapping):
                    if "total_loss" in outputs:
                        total_loss = outputs["total_loss"]
                    else:
                        # only key contains "loss" will be calculated.
                        total_loss = sum([v for k, v in outputs.items() if "loss" in k])
                        outputs["total_loss"] = total_loss
                elif isinstance(outputs, Sequence):
                    # list or tuple
                    total_loss = sum(outputs)
                else:
                    raise TypeError(f"unspported type of {outputs}: {type(outputs)}")

                if self.enable_amp:
                    self.grad_scaler.backward(self.grad_manager, total_loss)
                else:
                    self.grad_manager.backward(total_loss)

        if step_grad:
            if self.grad_clip_fn is not None:
                self.grad_clip_fn()
            self.optimizer.step().clear_grad()

        return outputs
