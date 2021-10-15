#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from abc import ABCMeta


class BaseEvaluator(metaclass=ABCMeta):

    def preprocess(self, input_data):
        return input_data

    def postprocess(self, model_outputs, input_data=None):
        return model_outputs

    def save_results(self, results):
        return results

    def evaluate(self, results):
        pass
