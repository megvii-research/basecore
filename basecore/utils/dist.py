#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import pickle

import megengine.distributed as dist

__all__ = [
    "init_local_pg",
    "get_local_pg",
    "get_local_world_size",
    "is_rank0_process",
    "all_reduce",
    "gather_pyobj",
]

_LOCAL_PROCESS_GROUP = None


def init_local_pg(rank_list):
    """init local process group"""
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None, "local process group already init"
    _LOCAL_PROCESS_GROUP = dist.new_group(rank_list)


def get_local_pg():
    return _LOCAL_PROCESS_GROUP


def get_local_world_size():
    assert _LOCAL_PROCESS_GROUP is not None
    return len(_LOCAL_PROCESS_GROUP.proc_ranks)


def is_rank0_process():
    """determine if process is rank0 process"""
    return dist.get_rank() == 0


def all_reduce(input_tensor, mode="sum"):
    """all reduce wrapper for dist.all_reduce

    Args:
        input_tensor (Tensor):
        mode (str): all_reduce mode, available mode: string in ["sum", "mean"] or upper case.
    """
    mode = mode.lower()
    assert mode in ["mean", "sum"]
    world_size = dist.get_world_size()
    if world_size == 1:
        return input_tensor
    else:
        sum_value = dist.functional.all_reduce_sum(input_tensor)
        ret_value = sum_value / world_size if mode == "mean" else sum_value
        return ret_value


def gather_pyobj(obj, obj_name, target_rank_id=0, reset_after_gather=True):
    """
    gather non tensor object into target rank.

    Args:
        obj (object): object to gather, for non python-buildin object, please
            make sure that it's picklable, otherwise gather process might be stucked.
        obj_name (str): name of pyobj, used for distributed client.
        target_rank_id (int): rank of target device. default: 0.
        reset_after_gather (bool): wheather reset value in client after get value. defualt: True.

    Returns:
        A list contains all objects if on target device, else None.
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return [obj]

    local_rank = dist.get_rank()
    client = dist.get_client()
    if local_rank == target_rank_id:
        obj_list = []
        for rank in range(world_size):
            if rank == target_rank_id:
                obj_list.append(obj)
            else:
                get_func = client.user_pop if reset_after_gather else client.user_get
                rank_data = get_func(f"{obj_name}{rank}")
                obj_list.append(pickle.loads(rank_data.data))

        return obj_list
    else:
        client.user_set(f"{obj_name}{local_rank}", pickle.dumps(obj))
