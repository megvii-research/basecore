#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import importlib


# https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
# or check https://docs.python.org/3/library/importlib.html
def import_module_with_path(module_path, module_name=None):
    if module_name is None:
        module_name = "tmp"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    import_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(import_module)
    return import_module


def import_content_from_path(content_name, module_path, module_name=None):
    import_module = import_module_with_path(module_path, module_name)
    content = getattr(import_module, content_name)
    return content
