#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import warnings
import megfile
import pickle

import megengine as mge

from .file import ensure_dir


class Checkpoint:

    def __init__(self, save_dir, model, tag_file="last_checkpoint", **data_with_state):
        """
        Args:
            save_dir (str): directory to save checkpoint. Support multi-backend.
            model (Module): saved module.
            tag_file (str): name of tag file. If tag file is None, no tag will be generated.
                Default value: "last_checkpoint".
            data_with_state: other class with `state_dict` method, such as optimizer
        """
        self.model = model
        self.save_dir = save_dir
        self.data_with_state = data_with_state
        self.tag_file = tag_file

    def save(self, save_name, save_with_tag=True, **extra_infos):
        """
        Save checkpoint as given name.

        Args:
            save_name (str): given name to save checkpoint.
            save_with_tag (bool): tag after saving or not, default: True.
            extra_infos (kwargs): other save states.
        """
        ckpt = {
            "model": self.model.state_dict(),
        }
        for key, obj in self.data_with_state.items():
            ckpt[key] = obj.state_dict()
        for key, value in extra_infos.items():
            ckpt[key] = value

        ensure_dir(self.save_dir)
        ckpt_file = megfile.smart_path_join(self.save_dir, save_name)
        with megfile.smart_open(ckpt_file, "wb") as f:
            mge.save(ckpt, f, pickle_protocol=pickle.DEFAULT_PROTOCOL)

        if save_with_tag and self.tag_file is not None:
            self.tag_checkpoint(ckpt_file)

    def get_checkpoint_file(self, filename=None):
        """
        Get file name of lastest checkpoint.

        Args:
            filename (str): If file name is None, use value in tag_file as filename instead.
        """
        if filename is not None:
            return megfile.smart_path_join(self.save_dir, filename)

        # get ckpt name from tag file
        try:
            tag_file = megfile.smart_path_join(self.save_dir, self.tag_file)
            with megfile.smart_open(tag_file, "r") as f:
                ckpt_name = f.read().strip()
        except Exception as e:
            if isinstance(e, TypeError):
                warnings.warn("please specify filename if tag_file of checkpoint is None")
            ckpt_name = ""

        return megfile.smart_path_join(self.save_dir, ckpt_name)

    def tag_checkpoint(self, file_name):
        """
        Args:
            file_name (str): checkpoint name with full path.
        """
        assert self.tag_file is not None, "tag checkpoint with invalid tag_file value"
        tag_file = megfile.smart_path_join(self.save_dir, self.tag_file)
        # TODO better basename method
        ckpt_name = os.path.basename(file_name)
        with megfile.smart_open(tag_file, "w") as f:
            f.write(ckpt_name)

    def resume(self, filename=None):
        """
        Resume checkpoint.

        Args:
            filename (str): checkpoint name. If filename is None, resume from file
                described in tag file. Defalult: None.
        """
        ckpt_name = self.get_checkpoint_file(filename)
        with megfile.smart_open(ckpt_name, "rb") as f:
            ckpt = mge.load(f)
        self.model.load_state_dict(ckpt["model"])
        for key, obj in self.data_with_state.items():
            obj.load_state_dict(ckpt[key])
        return ckpt

    def get(self, keys, default_value=None, filename=None):
        """
        Get value of given keys in checkpoint.

        Args:
            keys (str): key value to get value in checkpoint.
            default_value (str): Default value if keys is not found in ckptpoint.
            filename (str): checkpoint filename.
        """
        ckpt_name = self.get_checkpoint_file(filename)
        with megfile.smart_open(ckpt_name, "rb") as f:
            ckpt = mge.load(f)
        return ckpt.get(keys, default_value)
