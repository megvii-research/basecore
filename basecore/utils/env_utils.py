#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import socket
import subprocess
import sys
from tabulate import tabulate

import numpy as np

import megengine as mge

import basecore

__all__ = [
    "get_command_path",
    "get_device_info",
    "get_device_name",
    "get_env_info_table",
    "get_free_port",
    "get_hostip",
]


def get_device_count(device_type: str) -> int:
    """
    Make sure that `get_device_count` function is called correctly in BaseCore.
    """
    mge_ver = [int(x) for x in mge.__version__.split(".")[:2]]
    # breaking change since MGE 1.5.0
    if mge_ver >= [1, 5]:
        return mge.device.get_device_count(device_type)
    else:
        return mge.distributed.helper.get_device_count_by_fork(device_type)


def get_env_info_table(**kwargs):
    """Get environment infomation."""
    data = []

    data.append(("Python", sys.version))
    data.append(("Numpy", np.__version__))

    # image processing lib like cv2 and PIL
    try:
        import cv2
        data.append(("cv2", cv2.__version__))
    except ImportError:
        pass

    try:
        import PIL
        data.append(("Pillow", PIL.__version__))
    except ImportError:
        pass

    # Megvii related lib
    data.append(("MegEngine", mge.__version__))
    data.append(("basecore", basecore.__version__))

    # device info
    device_info_dict = get_device_info()
    for k, v in device_info_dict.items():
        data.append((k, v))

    # append extra infos in kwargs
    for k, v in kwargs.items():
        data.append((k, v))

    table_info = tabulate(data, tablefmt="fancy_grid")
    return table_info


def get_device_info():
    """
    Get device related information such as: host ip, num gpu, num cpu, cuda home, etc.
    """
    # device related
    info_dict = {}
    info_dict["sys.platform"] = sys.platform
    info_dict.update(get_hostip())

    num_cpus = get_device_count("cpu")
    info_dict["num cpus"] = num_cpus

    has_cuda = mge.is_cuda_available()
    info_dict["CUDA available"] = has_cuda

    if has_cuda:
        # CUDA related info
        CUDA_HOME = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
        info_dict["CUDA_HOME"] = CUDA_HOME

        num_gpus = get_device_count("gpu")
        key = "GPU " + ",".join([str(i) for i in range(num_gpus)])
        value = get_device_name()
        info_dict[key] = value

        try:
            nvcc = get_command_path("nvcc")
            nvcc = subprocess.check_output("'{}' -V | tail -n1".format(nvcc), shell=True)
            nvcc = nvcc.decode("utf-8").strip()
        except subprocess.SubprocessError:
            nvcc = "Not Available"
        finally:
            info_dict["NVCC"] = nvcc

    return info_dict


def get_command_path(command_name):
    """
    Get path of given command.

    NOTE: This function only works on linux platform.
    """
    with open(os.devnull, "w") as devnull:
        command_path = subprocess.check_output(
            ["which", command_name], stderr=devnull
        ).decode().rstrip('\r\n')
    return command_path


def get_device_name():
    """
    Get device name of GPU using nvidia-smi. A string like "GTX 2080 Ti" will be returned.
    """
    nvcc = get_command_path("nvidia-smi")
    nvcc_output = subprocess.check_output("{} -L".format(nvcc), shell=True).decode("utf-8")
    device_infos = nvcc_output.strip().split("\n")
    gpu_devices = [x.rsplit("(", maxsplit=1)[0].split(":")[-1].strip() for x in device_infos]
    return ",".join(set(gpu_devices))


# code are refered from:
# https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
def get_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def get_hostip_by_netiface():
    try:
        import netifaces
    except ImportError:
        return {}

    no_ip_string = "No IP addr"
    no_addr = [{"addr": no_ip_string}]
    ip_dict = {}

    for interface in netifaces.interfaces():
        # skip local
        if interface == "lo":
            continue
        net_dict = netifaces.ifaddresses(interface).setdefault(netifaces.AF_INET, no_addr)
        addr = [i["addr"] for i in net_dict]
        if len(addr) == 1:
            addr = addr[0]
        if addr == no_ip_string:
            continue
        ip_dict["IP of " + interface] = addr

    return ip_dict


def get_hostip():
    """
    Get host IP value. A dict contains IP information will be returned.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 10.255.255.255 is a local ip address, no need to be reachable
        host, port = "10.255.255.255", 1
        s.connect((host, port))
        IP = s.getsockname()[0]
    except Exception:
        # try by netiface
        ip_dict = get_hostip_by_netiface()
        if not ip_dict:
            IP = "127.0.0.1"
        else:
            return ip_dict
    finally:
        s.close()
    return {"IP": IP}
