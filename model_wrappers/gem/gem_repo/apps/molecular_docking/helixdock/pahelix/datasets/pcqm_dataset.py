#!/usr/bin/python
#-*-coding:utf-8-*-
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
Processing of PCQM dataset.

"""

import os
from os.path import join, exists
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from pahelix.datasets.inmemory_dataset import InMemoryDataset
USE_BL=bool(int(os.environ.get('USE_BL', '0')))
print('USE_BL:', USE_BL)


__all__ = ['get_default_pcqm_task_names', 'load_pcqm_dataset', 'get_pcqm_stat', 'get_pcqm_v2_stat', 'load_pcqm_v2_dataset']


def get_default_pcqm_task_names():
    """Get that default hiv task names and return class label"""
    return ['humolomo_gap']


def load_pcqm_dataset(data_path, task_names=None):
    """Load pcqm dataset,process the input information.
    tbd
    """
    if task_names is None:
        task_names = get_default_pcqm_task_names()

    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]

    data_list = []
    for i in range(len(smiles_list)):
        data = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data_list.append(data)
    dataset = InMemoryDataset(data_list)
    return dataset


def load_pcqm_v2_dataset(data_path, task_names='homolumogap'):
    """Load pcqm dataset,process the input information.
    tbd
    """
    if task_names is None:
        task_names = get_default_pcqm_task_names()

    raw_path = join(data_path, 'raw')
    sdf_path = join(data_path, 'pcqm4m-v2-train.sdf')
    csv_file = os.listdir(raw_path)[0]

    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]
    suppl = Chem.SDMolSupplier(sdf_path)
    # mols = [i for i in suppl]

    data_list = []
    # here we only count the labeled ones
    for i in tqdm(range(len(smiles_list))):
        data = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        if USE_BL and i < len(suppl):
            data['mol'] = suppl[i]
        # else:
        #     data['mol'] = None
        # data['mol'] = None
        data_list.append(data)
    dataset = InMemoryDataset(data_list)
    return dataset

def get_pcqm_stat(data_path, task_names):
    """Return mean and std of labels"""
    csv_file = join(data_path, 'raw/pcqm.csv')
    input_df = pd.read_csv(csv_file, sep=',')
    labels = input_df[task_names].dropna().values
    return {
        'mean': np.mean(labels, 0),
        'std': np.std(labels, 0),
        'N': len(labels),
    }

def get_pcqm_v2_stat(data_path, task_names):
    """Return mean and std of labels"""
    csv_file = join(data_path, 'raw/data.csv.gz')
    input_df = pd.read_csv(csv_file, sep=',')
    labels = input_df[task_names].dropna().values
    return {
        'mean': np.mean(labels, 0),
        'std': np.std(labels, 0),
        'N': len(labels),
    }