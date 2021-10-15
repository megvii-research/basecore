#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import weakref

from .hooks import BaseHook
from .progress import Progress
from .solver import Solver


class BaseTrainer:
    """
    Base class for epoch-wise trainer with hooks.

    Design of BaseTrainer is inspired by hook in detectron2, see more information on
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/train_loop.py

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        model: model used for training process.
        solver: solver used for training process. For megengine user,
            it contains optimzer and grad_manager at least.
        dataloader: data provider for training process.
        progress: object used for recording trainging processing.
    """

    def __init__(self, model, dataloader, solver: Solver, hooks=None):
        self.model = model
        self.dataloader = dataloader
        self.solver = solver

        # TODO wangfeng02: add hooks manager.
        self._hooks = []
        if hooks is not None:
            self.register_hooks(hooks)
        self.progress = Progress()

    def register_hooks(self, hooks):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, BaseHook)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train_in_epoch(self):
        for self.progress.epoch in range(self.progress.epoch, self.progress.max_epoch + 1):
            self.before_epoch()
            for self.progress.iter in range(1, self.progress.max_iter + 1):
                self.before_iter()
                self.train_one_iter()
                self.after_iter()
            self.after_epoch()

    def train(self, start_info, max_info):
        """
        Args:
            start_info (Iterable):
            max_info (Iterable):
        """
        # TODO use start_info and max_info as dict, assert them
        self.progress.update({
            "epoch": start_info[0], "iter": start_info[1],
            "max_epoch": max_info[0], "max_iter": max_info[1]
        })

        self.before_train()

        try:
            self.train_in_epoch()
        except Exception:
            raise

        # different from detectron2's design, `after_train` func is only executed
        # if training process runs without any exceptions. This design guarantees
        # all `after_train` funcs execute as expected.
        self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_epoch(self):
        for h in self._hooks:
            h.before_epoch()

    def after_epoch(self):
        for h in self._hooks:
            h.after_epoch()

    def before_iter(self):
        for h in self._hooks:
            h.before_iter()

    def after_iter(self):
        for h in self._hooks:
            h.after_iter()

    def train_one_iter(self):
        raise NotImplementedError
