#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
utils
"""

import sys
import os
from os.path import exists, dirname, basename, join
import numpy as np
import logging
import csv
from rdkit.ML.Scoring import Scoring


def exempt_parameters(src_list, ref_list):
    """Remove element from src_list that is in ref_list"""
    res = []
    for x in src_list:
        flag = True
        for y in ref_list:
            if x is y:
                flag = False
                break
        if flag:
            res.append(x)
    return res


def calc_parameter_size(parameter_list):
    """Calculate the total size of `parameter_list`"""
    count = 0
    for param in parameter_list:
        count += np.prod(param.shape)
    return count

def sequence_pad_2d(seq_list):
    """tbd"""
    seq_shape = []
    _tmp_seq_list = []
    for seq in seq_list:
        seq_shape.append(seq.shape)
        _tmp_seq_list.append(seq)
    seq_shape = np.array(seq_shape)
    max_value = np.max(seq_shape, axis=0, keepdims=True)

    max_len_1d = max_value[0][0]
    max_len_2d = max_value[0][1]
    print('[DEBUG] 1d and 2d : ', max_len_1d, ' ', max_len_2d)

    pad_seq_list = []
    for seq in _tmp_seq_list:
        print('[DEBUG] seq : ', seq)
        pad_seq = np.zeros([max_len_1d, max_len_2d])
        axi0, axi1 = seq.shape
        pad_seq[:axi0, :axi1] = seq
        pad_seq_list.append(pad_seq)
    pad_seqs = np.array(pad_seq_list)
    return pad_seqs


def sequence_pad(seq_list, max_len=None, pad_value=0):
    """tbd"""
    if max_len is None:
        max_len = np.max([len(seq) for seq in seq_list])

    pad_seq_list = []
    for seq in seq_list:
        if len(seq) < max_len:
            pad_shape = [max_len - len(seq)] + list(seq.shape[1:])
            pad_seq = np.concatenate([seq, np.full(pad_shape, pad_value, seq.dtype)], 0)
        else:
            pad_seq = seq[:max_len]
        pad_seq_list.append(pad_seq)

    pad_seqs = np.array(pad_seq_list)
    return pad_seqs


def sequence_mask(len_list, max_len=None):
    if max_len is None:
        max_len = np.max(len_list) 
    if max_len == 0:
        return np.zeros([len(len_list), 1], 'float32')
    seq_mask = np.zeros([len(len_list), max_len], 'float32')
    for i, l in enumerate(len_list):
        seq_mask[i, :l] = 1
    return seq_mask


def edge_to_pair(edge_index, edge_feat, max_len=None):
    edge_i, edge_j = edge_index[:, 0], edge_index[:, 1]
    if max_len is None:
        max_len = np.max(edge_i)
        max_len = max(np.max(edge_j), max_len) + 1
    pair_feat = np.zeros([max_len, max_len], edge_feat.dtype)
    pair_feat[edge_i, edge_j] = edge_feat
    return pair_feat


def pair_pad(pair_list, max_len=None, pad_value=0):
    """
    pair_list: [(n1, n1), (n2, n2), ...]
    return (B, N, N, *)
    """
    if max_len is None:
        max_len = np.max([len(x) for x in pair_list])    

    pad_pair_list = []
    for pair in pair_list:
        raw_shape = list(pair.shape)
        if raw_shape[0] < max_len:
            max_shape = [max_len, max_len] + raw_shape[2:]
            pad_width = [(0, x - y) for x, y in zip(max_shape, raw_shape)]
            pad_pair = np.pad(pair, pad_width, 'constant', constant_values=pad_value)
        else:
            pad_pair = pair[:max_len, :max_len]
        pad_pair_list.append(pad_pair)
    return np.array(pad_pair_list)  # (B, N, N, *)


def tree_map(f, d):
    new_d = {}
    for k in d:
        if type(d[k]) is dict:
            new_d[k] = tree_map(f, d[k])
        else:
            new_d[k] = f(d[k])
    return new_d


def tree_flatten(d):
    new_d = {}
    for k in d:
        if type(d[k]) is dict:
            cur_d = tree_flatten(d[k])
            for sub_k, sub_v in cur_d.items():
                new_d[f'{k}.{sub_k}'] = sub_v
        else:
            new_d[k] = d[k]
    return new_d


def tree_filter(key_cond, value_cond, d):
    new_d = {}
    for k in d:
        if not key_cond is None and not key_cond(k):
            continue
        if not value_cond is None and not value_cond(d[k]):
            continue

        if type(d[k]) is dict:
            cur_d = tree_filter(key_cond, value_cond, d[k])
            if len(cur_d) != 0:
                new_d[k] = cur_d
        else:
            new_d[k] = d[k]
    return new_d


def add_to_data_writer(data_writer, step, results, prefix=''):
    """tbd"""
    logging.info(f"step:{step} {prefix} {results}")
    if data_writer is None:
        return
    for k, v in results.items():
        data_writer.add_scalar(prefix + k, v, step)


def write_to_csv(csv_name, value_dict):
    """
    tbd
    """
    with open(csv_name, 'a') as f:
        csv_writer = csv.DictWriter(f, fieldnames=list(value_dict.keys()))
        if value_dict['epoch'] == 0:
            csv_writer.writeheader()
        csv_writer.writerow(value_dict) 


def set_logging_level(level):
    level_dict = {
        "NOTSET": logging.NOTSET,
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s',
        level=level_dict[level],
        datefmt='%Y-%m-%d %H:%M:%S')

