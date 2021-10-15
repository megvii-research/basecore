#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.


class BaseHook:
    """
    Base class for hooks that can be registered with :class:`BaseTrainer`.

    Design of BaseHook is inspired by hook in detectron2, see more information on
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/train_loop.py

    Each hook could implement 8 methods. The way they are called is demonstrated
    in the following snippet:

    .. code-block:: python

        hook.before_train()
        for epoch in range(start_epoch, max_epoch):
            hook.before_epoch()
            for iter in range(start_iter, max_iter):
                hook.before_iter()
                trainer.run_iter()
                hook.after_iter()
            hook.after_epoch()
        hook.after_train()

    Notes:
        1. In the hook method, users can access `self.trainer` to access more
           properties about the context (e.g., current iteration, dataloader, model).
        2. User could use hook to transport information from one object to anohter.

    Attributes:
        trainer: A weak reference to the trainer object. Set by the trainer
            when the hook is registered.
    """

    def before_train(self):
        """
        Called before training process.
        """
        pass

    def after_train(self):
        """
        Called after training process.
        """
        pass

    def before_epoch(self):
        """
        Called before each epoch.
        """
        pass

    def after_epoch(self):
        """
        Called after each epoch.
        """
        pass

    def before_iter(self):
        """
        Called before each iteration.
        """
        pass

    def after_iter(self):
        """
        Called after each iteration.
        """
        pass
