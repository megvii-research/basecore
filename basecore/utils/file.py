#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import megfile

__all__ = ["ensure_dir"]


def ensure_dir(path: str):
    """create directories if *path* does not exist"""
    if not megfile.smart_isdir(path):
        megfile.smart_makedirs(path, exist_ok=True)
