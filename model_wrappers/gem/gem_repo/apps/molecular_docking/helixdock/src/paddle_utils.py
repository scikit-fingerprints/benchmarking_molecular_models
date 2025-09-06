#!/usr/bin/python                                                                                                
#-*-coding:utf-8-*- 
#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
paddle utils
"""

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed.fleet.utils import recompute
import pgl


def dist_all_reduce(x, *args, **argv):
    """
    make dist.all_reduce returnable.
    x: tensor
    """
    dist.all_reduce(x, *args, **argv)
    return x


def dist_mean(array, distributed=False):
    """tbd"""
    n = len(array)
    x_sum = 0 if n == 0 else np.sum(array)
    if distributed:
        n = dist_all_reduce(paddle.to_tensor(n, dtype='int64')).numpy()[0]
        x_sum = dist_all_reduce(paddle.to_tensor(x_sum, dtype='float32')).numpy()[0]
    x_mean = 0 if n == 0 else x_sum / n
    return x_mean


def dist_sum(array, distributed=False):
    """tbd"""
    n = len(array)
    x_sum = 0 if n == 0 else np.sum(array)
    if distributed:
        x_sum = dist_all_reduce(paddle.to_tensor(x_sum, dtype='float32')).numpy()[0]
    return x_sum


def dist_length(array, distributed=False):
    """tbd"""
    n = len(array)
    if distributed:
        n = dist_all_reduce(paddle.to_tensor(n, dtype='int64')).numpy()[0]
    return n


def dist_all_gather_object(obj):
    """tbd"""
    obj_list = []
    dist.all_gather_object(obj_list, obj)
    return obj_list


def recompute_wrapper(func, *args, is_recompute=True):
    """tbd"""
    if is_recompute:
        return recompute(func, *args)
    else:
        return func(*args)
        

def maybe_to_tensor(x):
    if isinstance(x, np.ndarray):
        return paddle.to_tensor(x)
    elif isinstance(x, pgl.Graph):
        return x.tensor()
    else:
        return x
