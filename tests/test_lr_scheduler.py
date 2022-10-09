#!/usr/bin/env python3

import unittest

import numpy as np

from megengine.module import Linear
from megengine.optimizer import SGD

from basecore.engine import (
    ChainedScheduler,
    CosineAnnealingLR,
    CyclicLR,
    LinearLR,
    MultiStepLR,
    StepLR,
    WarmUpScheduler
)


class LRSchedulerTest(unittest.TestCase):

    def setUp(self):
        module = Linear(3, 3)
        self.lr = 0.1
        self.opt = SGD(module.parameters(), lr=self.lr)

    def test_warmup_lr_update(self):
        sche = StepLR(self.opt, step_size=8)
        sche = WarmUpScheduler(sche, warmup_length=5)
        repr(sche)  # test repr method
        factor = [0.2, 0.4, 0.6, 0.8, 1, 1, 1, 1]
        gt_lr = [self.lr * f for f in factor]

        for step_count, lr in enumerate(gt_lr):
            self.assertEqual(step_count, sche.scheduler._step_count)
            sche.step()
            sche_lr = sche.get_last_lr()[0]
            self.assertTrue(
                np.allclose(lr, sche_lr),
                msg=f"fail at step {step_count}: lr {lr}, gt {sche_lr}",
            )

    def test_step_lr_update(self):
        size = 3
        sche = StepLR(self.opt, step_size=size, gamma=0.1)
        repr(sche)  # test repr method
        gt_lrs = [self.lr] * size + [self.lr * 0.1] * size + [self.lr * 0.01]

        for step_count, lr in enumerate(gt_lrs):
            sche.step()
            sche_lr = sche.get_last_lr()[0]
            self.assertTrue(
                np.allclose(lr, sche_lr),
                msg=f"fail at step {step_count}: lr {lr}, gt {sche_lr}",
            )

    def test_multistep_lr(self):
        sche = MultiStepLR(self.opt, milestones=[2, 5, 7])
        repr(sche)  # test repr method
        gt_lrs = [self.lr] * 2 + [self.lr * 0.1] * 3 + [self.lr * 0.01] * 2 + [self.lr * 0.001] * 2

        for step_count, lr in enumerate(gt_lrs):
            sche.step()
            sche_lr = sche.get_last_lr()[0]
            self.assertTrue(
                np.allclose(lr, sche_lr),
                msg=f"fail at step {step_count}: lr {lr}, gt {sche_lr}",
            )

    def test_linear_lr(self):
        sche = LinearLR(self.opt, start_factor=0.5, total_iters=6)
        repr(sche)  # test repr method
        gt_lrs = [0.01 * x for x in range(5, 11)] + [0.1] * 2

        for step_count, lr in enumerate(gt_lrs):
            sche.step()
            sche_lr = sche.get_last_lr()[0]
            self.assertTrue(
                np.allclose(lr, sche_lr),
                msg=f"fail at step {step_count}: lr {lr}, gt {sche_lr}",
            )

    def test_cosine_lr(self):
        eta_min = 0.05
        period = 5
        prev_lr = self.lr
        sche = CosineAnnealingLR(self.opt, period=period, eta_min=eta_min)
        for step_count in range(2 * period):
            sche.step()
            lr = sche.get_last_lr()[0]
            if step_count == 0:
                self.assertEqual(self.lr, lr)
            elif step_count < period:
                self.assertTrue(lr < self.lr)
                self.assertTrue(lr > eta_min)
                self.assertTrue(lr < prev_lr)
            elif step_count == 5:
                self.assertEqual(eta_min, lr)
            else:
                self.assertTrue(lr < self.lr)
                self.assertTrue(lr > eta_min)
                self.assertTrue(lr > prev_lr)
            prev_lr = lr

    def test_cyclic_lr(self):
        base_lr = 0.05
        sche = CyclicLR(self.opt, base_lr=base_lr, max_lr=1.0)
        repr(sche)  # test repr method
        delta = 0.000475
        gt_lrs = [base_lr + delta * i for i in range(10)]

        for step_count, lr in enumerate(gt_lrs):
            sche.step()
            sche_lr = sche.get_last_lr()[0]
            self.assertTrue(
                np.allclose(lr, sche_lr),
                msg=f"fail at step {step_count}: lr {lr}, gt {sche_lr}",
            )

    def test_chained_lr(self):
        sche1 = LinearLR(self.opt, start_factor=0.4, total_iters=2)
        sche2 = MultiStepLR(self.opt, [2, 5])
        sche = ChainedScheduler([sche1, sche2], switch_steps=[3])
        repr(sche)  # test repr method

        gt_lrs = [0.04, 0.1, 0.1] + [0.1, 0.1, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001]

        for step_count, lr in enumerate(gt_lrs):
            sche.step()
            sche_lr = sche.get_last_lr()[0]
            self.assertTrue(
                np.allclose(lr, sche_lr),
                msg=f"fail at step {step_count}: lr {lr}, gt {sche_lr}",
            )


if __name__ == "__main__":
    unittest.main()
