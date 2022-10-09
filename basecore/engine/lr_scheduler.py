#!/usr/bin/env python3
import math
from typing import List

from megengine.optimizer import Optimizer

__all__ = [
    "LRScheduler",
    "CosineAnnealingLR",
    "CyclicLR",
    "ChainedScheduler",
    "WarmUpScheduler",
    "StepLR",
    "MultiStepLR",
    "LinearLR",
]


class LRScheduler:

    def __init__(self, optimizer):
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        # Initialize base learning rates
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._step_count = 0  # step_count value is 0 during the first step when `get_lr` is called

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """Return last computed learning rate by current scheduler. Usually used for logging"""
        return self._last_lr

    def get_lr(self, *args):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def step(self, *args):
        values = self.get_lr(*args)
        self.set_lr(values)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        self._step_count += 1

    def set_lr(self, values):
        # set lr for optimzer
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group["lr"] = lr

    def __repr__(self) -> str:
        optimizer_name = type(self.optimizer).__name__
        return f"{self.__class__.__name__}(optimizer={optimizer_name})"


class ChainedScheduler(LRScheduler):
    """
    ChainedScheduler is a simple wrapper to concatenate multiple schedulers.
    For example, if you want to use CosineAnnealingLR scheduler for the first 10 iters,
    and after 10 iters, using StepLR scheduler.

    Args:
        schedulers (List[LRScheduler]): list of schedulers.
        switch_steps (List[int]): switch steps. The length of switch_steps
            should be the same as len(schedulers) - 1.

    Example:
        >>> # Assuming optimizer uses lr = 0.1 for all groups
        >>> # lr = 0.04   if iter == 0
        >>> # lr = 0.1    if 1 <= iter <= 2
        >>> # switch to MultiStepLR scheduler
        >>> # lr = 0.1    if 3 <= iter <= 4
        >>> # lr = 0.01   if 5 <= iter <= 7
        >>> # lr = 0.001  if iter > 7
        >>> linear = LinearLR(optimizer, start_factor=0.4, total_iters=2)
        >>> step = MultiStepLR(optimizer, [2, 5])
        >>> scheduler = ChainedSchedulers([linear, step], switch_steps=[3])
        >>> for _ in range(10):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, schedulers: List[LRScheduler], switch_steps: List[int]):
        assert len(schedulers) > 1, "Scheduler list should have at least 2 schedulers."
        if len(schedulers) != len(switch_steps) + 1:
            raise ValueError(
                "Number of schedulers should be equal to number of switch steps + 1."
            )
        self.schedulers = schedulers
        self.cur_scheduler = schedulers[0]
        super(ChainedScheduler, self).__init__(self.cur_scheduler.optimizer)
        self.switch_steps = switch_steps

    def step(self, *args):
        if self._step_count in self.switch_steps:
            # switch to next scheduler
            self.cur_scheduler = self.schedulers[self.switch_steps.index(self._step_count) + 1]

        self.cur_scheduler.step(*args)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        self._step_count += 1


class MultiSchedulers(LRScheduler):
    """
    MultiScheduler is a simple wrapper for multiple schedulers used to combine multiple schedulers.
    For example, if backbone and head share different learning rate decay strategy,
    this class can be used to combine the two lrscheduler.

    Example:
        >>> backbone_scheduler = CosineAnnealingLR(optimizer, 100)
        >>> head_scheduler = StepLR(optimizer, 100)
        >>> schedulers = MultiSchedulers([backbone_scheduler, head_scheduler])
        >>> for _ in range(100):
        >>>     schedulers.step()
        >>>     train(...)
        >>>     validate(...)
    """
    def __init__(self, schedulers) -> None:
        assert len(schedulers) > 1, "Scheduler list should have at least 2 schedulers."
        # TODO: check if all schedulers have reused parameters
        self._step_count = 0

    def get_last_lr(self):
        """Return last computed learning rate by current scheduler. Usually used for logging"""
        last_lr = []
        for scheduler in self.schedulers:
            last_lr.extend(scheduler.get_last_lr())
        return last_lr

    def step(self, *args):
        for scheduler in self.schedulers:
            scheduler.step(*args)
        self._step_count += 1


class WarmUpScheduler(LRScheduler):
    """
    WarmUpScheduler is a wrapper class of LRScheduler.
    It add functionality of warmup learning rate.

    NOTE: WarmUpScheduler will override the original learning rate.

    Args:
        scheduler (Scheduler): LRScheduler to wrap.
        warmup_length (int): warmup length. For iterwise scheduler, it is the number of iterations.
            For epochwise scheduler, it is the number of epochs.
        warmup_method (str): warmup method. Default: linear.
        lr_factor (float): factor of lr, make sense only using constant warmup method. default to None.  # noqa

    Example:
        >>> # Assuming optimizer uses lr = 0.1 for all groups
        >>> # lr = 0.02   if iter == 0
        >>> # lr = 0.04   if iter == 1
        >>> # lr = 0.06   if iter == 2
        >>> # lr = 0.08   if iter == 3
        >>> # lr = 0.1    if iter >= 4
        >>> scheduler = ConstantLR(self.opt)
        >>> scheduler = WarmUpScheduler(scheduler, warmup_length=5)
        >>> for _ in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(
        self,
        scheduler: LRScheduler,
        warmup_length: int,
        warmup_method: str = "linear",
        lr_factor: float = None,
    ):
        self.scheduler = scheduler

        assert warmup_method in ["linear", "constant"], f"Unknown warmup method: {warmup_method}"
        self.warmup_method = warmup_method
        self.warmup_length = warmup_length
        if warmup_method == "constant":
            if lr_factor is None:
                raise ValueError("lr_factor must be specified when using constant warmup method")
        self.lr_factor = lr_factor

    def warmup_lr(self, count):
        if self.warmup_method == "linear":
            lr_factor = float(count / self.warmup_length)
        elif self.warmup_method == "constant":
            lr_factor = self.lr_factor
        return [lr_factor * lr for lr in self.base_lrs]

    def step(self, *args):
        values = self.get_lr(*args)
        self.set_lr(values)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        self.scheduler._step_count += 1

    def get_lr(self, *args):
        step_count = self.scheduler._step_count
        if step_count < self.warmup_length:
            lr = self.warmup_lr(step_count + 1)
        else:
            lr = self.scheduler.get_lr(*args)
        return lr

    def __repr__(self):
        return f"WarmUpScheduler({repr(self.scheduler)}, " \
            f"warmup_length={self.warmup_length}, " \
            f"warmup_method={self.warmup_method})"

    def __getattr__(self, name):
        # if attribute could not be found in WarmUpScheduler,
        # try to find it in the wrapped scheduler
        if name not in self.__dict__:
            return getattr(self.scheduler, name)
        return self.__dict__[name]

    def __setattr__(self, __name: str, __value) -> None:
        if __name in ["base_lrs", "_step_count"]:
            self.scheduler.__setattr__(__name, __value)
        return super().__setattr__(__name, __value)


class StepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.

    Example:
        >>> # Assuming optimizer uses lr = 0.01 for all groups
        >>> # lr = 0.01     if epoch < 30
        >>> # lr = 0.001    if 30 <= epoch < 60
        >>> # lr = 0.0001   if 60 <= epoch < 90
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, step_size, gamma=0.1):
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer)

    def get_lr(self, *args):
        step_count = self._step_count
        if (step_count == 0) or (step_count % self.step_size != 0):
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]


class MultiStepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, milestones, gamma=0.1):
        self.milestones = list(sorted(set(milestones)))
        self.gamma = gamma
        super(MultiStepLR, self).__init__(optimizer)

    def get_lr(self, *args):
        step_count = self._step_count
        if step_count not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        else:
            return [group["lr"] * self.gamma for group in self.optimizer.param_groups]


class LinearLR(LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 0.0.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.

    Example:
        >>> # Assuming optimizer uses lr = 0.1 for all groups
        >>> # lr = 0.05   if epoch == 0
        >>> # lr = 0.06   if epoch == 1
        >>> # lr = 0.07   if epoch == 2
        >>> # lr = 0.08   if epoch == 3
        >>> # lr = 0.09   if epoch == 4
        >>> # lr = 0.1    if epoch >= 5
        >>> scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=6)
        >>> for epoch in range(10):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, start_factor=0.0, end_factor=1.0, total_iters=5):
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError('Starting multiplicative factor expected to be between 0 and 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError('Ending multiplicative factor expected to be between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.factor_per_step = (end_factor - start_factor) / (total_iters - 1)
        self.total_iters = total_iters
        super(LinearLR, self).__init__(optimizer)

    def get_lr(self, *args):
        step_count = self._step_count

        if step_count >= self.total_iters:
            return self.base_lrs

        factor = self.start_factor + step_count * self.factor_per_step
        return [lr * factor for lr in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    It has been proposed in `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983

    Notice that because the schedule is defined recursively, the learning rate
    can be simultaneously modified outside this scheduler by other operators.
    If the learning rate is set solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        period (int): peroid of cosine annealing. Usually it is the number of iterations.
        eta_min (float): Minimum eta value. Default: 0.0.
    """

    def __init__(self, optimizer, period: int, eta_min: float = 0.0):
        self.period = period
        self.eta_min = eta_min
        super(CosineAnnealingLR, self).__init__(optimizer)
        self.scalars = [(eta_max - eta_min) / 2.0 for eta_max in self.base_lrs]

    def get_lr(self, *args):
        step_count = self._step_count
        factor = 1 + math.cos(math.pi * step_count / self.period)
        return [self.eta_min + s * factor for s in self.scalars]


class CyclicLR(LRScheduler):
    r"""Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.
    This class has three built-in policies, as put forth in the paper:
    * "triangular": A basic triangular cycle without amplitude scaling.
    * "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle.
    * "exp_range": A cycle that scales initial amplitude by
        :math:`\text{gamma}^{\text{cycle iterations}}` at each cycle iteration.

    This implementation was adapted from the pytorch implementation.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for each parameter group.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size_up (int): Number of training iterations in the
            increasing half of a cycle. Default: 2000
        step_size_down (int): Number of training iterations in the
            decreasing half of a cycle. If step_size_down is None,
            it is set to step_size_up. Default: None
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            If specified, then 'mode' is ignored.
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.8
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            The momentum at any cycle is the difference of max_momentum
            and some scaling of the amplitude; therefore
            base_momentum may not actually be reached depending on
            scaling function. Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.9

    Example:
        >>> scheduler = CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
        >>> for epoch in range(10):
        >>>     scheduler.step()
        >>>     train_step(...)
        >>>     val_step(...)

    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(
        self, optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None,
        mode="triangular", gamma=1.0, scale_fn=None, scale_mode="cycle",
        cycle_momentum=True, base_momentum=0.8, max_momentum=0.9,
    ):
        super(CyclicLR, self).__init__(optimizer)

        group_length = len(optimizer.param_groups)
        f = lambda x: x if isinstance(x, (list, tuple)) else [x] * group_length  # noqa
        self.base_lrs = f(base_lr)
        self.max_lrs = f(max_lr)
        self.base_momentums = f(base_momentum)
        self.max_momentums = f(max_momentum)

        step_size_up = float(step_size_up)
        step_size_down = step_size_up if step_size_down is None else float(step_size_down)
        self.total_size = step_size_up + step_size_down
        self.step_ratio = step_size_up / self.total_size

        # set scale_fn and scale_mode
        if scale_fn is None:
            if mode not in ["triangular", "triangular2", "exp_range"] and scale_fn is None:
                raise ValueError("mode is invalid and scale_fn is None")
            if mode == "triangular":
                self.scale_fn = lambda x: 1.0
                self.scale_mode = "cycle"
            elif mode == "triangular2":
                self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
                self.scale_mode = "cycle"
            elif mode == "exp_range":
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = "iterations"
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if "momentum" not in optimizer.param_groups[0]:
                raise ValueError("optimizer must support momentum when enable `cycle_momentum`")
            for momentum, group in zip(self.base_momentums, optimizer.param_groups):
                group["momentum"] = momentum

    def get_lr(self, *args):
        """
        Calculates the learning rate at batch index. This function treats `self._step_count`
        as the last batch index.

        NOTE: If `self.cycle_momentum` is ``True``, this function has a side effect
            of updating the optimizer's momentum.
        """
        step_count = self._step_count
        cycle = math.floor(1 + step_count / self.total_size)
        x = 1. + step_count / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        scale_x = cycle if self.scale_mode == "cycle" else step_count
        scale = scale_factor * self.scale_fn(scale_x)
        lrs = [
            base_lr + (max_lr - base_lr) * scale
            for base_lr, max_lr in zip(self.base_lrs, self.max_lrs)
        ]

        if self.cycle_momentum:
            momentums = [
                max_m - (max_m - base_m) * scale
                for base_m, max_m in zip(self.base_momentums, self.max_momentums)
            ]
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group["momentum"] = momentum

        return lrs
