#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import contextlib
import datetime
import io
import logging
import os
from loguru import logger

from .inspector import get_call_count, get_last_call_deltatime

"""
log_* function in this file are mainly inspired by
https://github.com/abseil/abseil-py/blob/master/absl/logging/__init__.py
"""

__all__ = [
    "str_timestamp",
    "log_every_n_seconds",
    "log_every_n_calls",
    "log_first_n_calls",
    "redirect_to_loguru",
    "redirect_mge_logger_to_loguru",
    "setup_mge_logger",
]


def str_timestamp(time_value=None):
    """format given timestamp, if no timestamp is given, return a call time string"""
    if time_value is None:
        time_value = datetime.datetime.now()
    return time_value.strftime("%Y-%m-%d_%H-%M-%S")


def log_every_n_seconds(msg, n=1, log_level="INFO"):
    """
    Log message per n seconds.

    Args:
        log_level (int): the logging level
        msg (str): logging message.
        n (int): n seconds value.
    """
    delta_time = get_last_call_deltatime(reset=False, reset_seconds=n)
    if delta_time is None or delta_time >= n:
        logger.opt(depth=1).log(log_level, msg)


def log_every_n_calls(msg, n=1, log_level="INFO"):
    """
    Log message per n calls.

    Args:
        msg (str): logging message.
        n (int): n value.
        log_level (int): the logging level.
    """
    call_count = get_call_count()
    if call_count % n == 0:
        logger.opt(depth=1).log(log_level, msg)


def log_first_n_calls(msg, n=5, log_level="INFO"):
    """
    Log message during first n calls.

    Args:
        msg (str): logging message.
        n (int): n value.
        log_level (int): the logging level.
    """
    call_count = get_call_count()
    print(call_count)
    if call_count <= n:
        logger.opt(depth=1).log(log_level, msg)


@contextlib.contextmanager
def redirect_to_loguru(log_level="INFO", depth=2):
    """
    redirect output to io buffer and use loguru to log.

    Args:
        log_level (str): the logging level.
        depth (int): logger stack depth, using 2 to log at caller level.
    """
    io_buffer = io.StringIO()
    with contextlib.redirect_stdout(io_buffer):
        yield
    logger.opt(depth=depth).log(log_level, "\n" + io_buffer.getvalue())


class InterceptHandler(logging.Handler):

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def redirect_mge_logger_to_loguru():
    from megengine.logger import _all_loggers as all_mge_loggers

    intercept_hdlr = InterceptHandler()
    for mge_logger in all_mge_loggers:
        remove_hdlrs = [
            hdlr for hdlr in mge_logger.handlers
            if isinstance(hdlr, logging.StreamHandler)
        ]
        for hdlr in remove_hdlrs:
            mge_logger.removeHandler(hdlr)
        mge_logger.addHandler(intercept_hdlr)


def setup_mge_logger(path=None, log_level="INFO", to_loguru=False):
    from .dist import is_rank0_process
    import megengine.logger as mge_logger

    def generate_log_path(filename):
        return os.path.join(path, filename) if path is not None else filename

    if to_loguru:
        redirect_mge_logger_to_loguru()
    mgb_logger = mge_logger.get_logger("megbrain")
    # remove StreamHandler for Megbrain logger
    del mgb_logger.handlers[:]
    mgb_log_path = generate_log_path("mgb_log.txt")
    mgb_logger.addHandler(logging.FileHandler(filename=mgb_log_path))

    if is_rank0_process():
        logger.info("Megbrain logging info will be redirected into {}".format(mgb_log_path))
        mge_logger.set_log_file(generate_log_path("mge_log.txt"))
        mge_logger.set_log_level(log_level)
    else:
        mge_logger.set_log_level("ERROR")
