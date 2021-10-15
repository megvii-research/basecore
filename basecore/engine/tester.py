#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import datetime
import time
from loguru import logger

import megengine as mge
import megengine.distributed as dist

from basecore.network import adjust_stats
from basecore.utils import gather_pyobj, is_rank0_process, log_every_n_seconds


class BaseTester:
    """
    Base class for iterative trainer with hooks.
    core logic is implemented in function `test`.
    """

    def __init__(self, model, dataloader, evaluator=None):
        self.model = model
        self.dataloader = dataloader
        self.evaluator = evaluator

    def inference(self, warm_iters=5, log_seconds=5):
        """inference model on dataloader.

        Args:
            warm_iters (int): number of warmup iters, defalut: 5.
            log_second (int): log time interval in second. default: 5.
        """
        total_iters = len(self.dataloader)
        warm_iters = min(warm_iters, total_iters)

        results_list = []
        total_time = 0
        with adjust_stats(self.model, training=False) as model:
            for iters, data in enumerate(self.dataloader, 1):
                if iters == warm_iters + 1:
                    total_time = 0

                model_inputs = self.preprocess_inputs(data)
                start_time = time.perf_counter()
                net_outputs = model(model_inputs)
                # use full_sync func to get more precise time
                mge._full_sync()
                total_time += time.perf_counter() - start_time
                results_list.append(self.postprocess_outputs(net_outputs, data))

                count_iters = iters - warm_iters if iters > warm_iters else iters
                time_per_iter = total_time / count_iters
                infer_eta = (total_iters - iters) * time_per_iter
                log_every_n_seconds(
                    "Inference process {}/{}, average speed:{:.4f}s/iters. ETA:{}".format(
                        iters, total_iters, time_per_iter,
                        datetime.timedelta(seconds=int(infer_eta))
                    ),
                    n=log_seconds,
                )
            logger.info(
                "Finish inference process, total time:{}, average speed:{:.4f}s/iters.".format(
                    datetime.timedelta(seconds=int(total_time)),
                    total_time / (len(self.dataloader) - warm_iters),
                )
            )
            return self.gather_inference_results(results_list)

    def gather_inference_results(self, results):
        """
        Gather all inference result to master rank.
        Used in last process of `inference` function of Tester.
        """
        results_list = gather_pyobj(results, "obj_rank", target_rank_id=0, reset_after_gather=True)
        logger.info("Gather all results to rank0 device.")
        if is_rank0_process():
            gather_results = []
            for res in results_list:
                gather_results.extend(res)
            return gather_results

    def preprocess_inputs(self, input_data):
        """
        Proprocess inputs to fit given model inputs, override this function for your own test logic.
        Used in `inference` function of Tester.

        Args:
            input_data: data provided by test dataloader.

        Returns:
            return value should be a Tensor.
        """
        if self.evaluator is not None:
            return self.evaluator.preprocess(input_data)
        return mge.Tensor(input_data)

    def postprocess_outputs(self, results, data):
        """
        Postprocess to fit given model inputs, override this function for your own test logic.
        Used in `inference` function of Tester.

        Args:
            results: model inference results.
            data: input data, usually used to provide image info like (height, width).

        Returns:
            return value should be a Tensor by default. Or, user should
                override `gather_inference_reuslts` function.
        """
        if self.evaluator is not None:
            return self.evaluator.postprocess(results, data)
        return results

    def save_results(self, results):
        """
        save results into filename, this function highly depends on evaluator of dataset.
        if evaluator is not given, results will not be saved by default.
        """
        if self.evaluator is not None:
            return self.evaluator.save_results(results)
        return results

    def evaluate(self, results):
        """
        evaluate module using results, if evaluator is not given, evaluate process will be passed.
        override this function if you don't want evaluator.

        Args:
            results: inference results, could be any type.
        """
        if self.evaluate is not None:
            self.evaluator.evaluate(results)

    def test(self):
        results = self.inference()
        # only evaluate result on main process
        if is_rank0_process():
            results = self.save_results(results)
            self.evaluate(results)
        if dist.get_world_size() > 1:
            dist.group_barrier()

    def resume(self, checkpoint):
        # TODO wangfeng02: thinking of support such an tester.
        pass
