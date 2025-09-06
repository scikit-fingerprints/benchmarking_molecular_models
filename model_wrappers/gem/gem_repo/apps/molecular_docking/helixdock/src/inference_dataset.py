#!/usr/bin/env python3
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
dataset
"""

from cProfile import label
import os
from os.path import join, exists, basename, dirname
import time
import re
import logging
from glob import glob
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd

from rdkit import Chem
import random

# import torch
import paddle
import paddle.distributed as dist
from pahelix.utils.basic_utils import mp_pool_map
from pahelix.utils.compound_tools import new_mol_to_graph_data
from pahelix.utils.data_utils import save_data_list_to_npz, load_npz_to_data_list
from pahelix.datasets.inmemory_dataset import InMemoryDataset

from .utils import tree_map
from .mol_utils import load_ligand, load_protein, extract_pocket, \
        extract_pocket_from_box, get_bond_length, generate_ETKDG_conformer


def ligand_pocket_data_to_docking_data(data_lig, data_poc):
    """
    `atom_pos` is real atom pos, while `zero_atom_pos` and 
    `ETKDG_atom_pos` are generated atom pos.
    """
    assert 'atom_pos' in data_lig
    assert 'atom_pos' in data_poc
    ligand_atom_pos = data_lig['atom_pos']
    protein_atom_pos = data_poc['atom_pos']
    protein_center = np.mean(protein_atom_pos, 0, keepdims=True)
    ligand_zero_atom_pos = np.tile(protein_center, [len(ligand_atom_pos), 1])

    if 'ETKDG_atom_pos' in data_lig:
        ligand_ETKDG_atom_pos = data_lig['ETKDG_atom_pos']
        ligand_ETKDG_atom_pos = ligand_ETKDG_atom_pos - \
                np.mean(ligand_ETKDG_atom_pos, 0, keepdims=True) + protein_center
        data_lig['bond_length'] = get_bond_length(ligand_ETKDG_atom_pos, data_lig['edges'])
    else:
        data_lig['bond_length'] = get_bond_length(ligand_zero_atom_pos, data_lig['edges'])
    
    data_poc['bond_length'] = get_bond_length(protein_atom_pos, data_poc['edges'])

    data = {}
    data.update({f'ligand_{k}': v for k, v in data_lig.items()})
    data.update({f'protein_{k}': v for k, v in data_poc.items()})
    data['ligand_zero_atom_pos'] = ligand_zero_atom_pos
    return data

class InferenceDataset(paddle.io.Dataset):
    """tbd."""
    def __init__(self, 
            dataset_config, model_config, encoder_config,
            trainer_id=0, trainer_num=1, num_workers=8):
        self.dataset_config = dataset_config
        self.data_config = model_config.data
        self.trainer_id = trainer_id
        self.trainer_num = trainer_num
        self.num_workers = max(1, num_workers)

        done_file = join(self.dataset_config.cache_dir, '.done')
        if not exists(done_file):
            logging.info('[InferenceDataset] load raw dataset')
            dataset = self._load_dataset(trainer_id, trainer_num)
            logging.info(f'[InferenceDataset] transform raw dataset: {len(dataset)}')
            rank_done_file = f'{self.dataset_config.cache_dir}/rank-{trainer_id}/.done'
            if not os.path.exists(rank_done_file):
                dataset.transform(self.get_sample, num_workers=self.num_workers, drop_none=True)
                logging.info(f'[InferenceDataset] save dataset: {len(dataset)}')
                dataset.save_data(f'{self.dataset_config.cache_dir}/rank-{trainer_id}')
                logging.info(f'[InferenceDataset] save dataset done: {len(dataset)}')
                open(rank_done_file, 'w').write('')
            else:
                logging.info(f'[InferenceDataset] save dataset already done: {len(dataset)}')
            if trainer_num > 1:
                logging.info(f'rank-{trainer_id} start waiting')
                dist.barrier()
            if trainer_id == 0:
                open(done_file, 'w').write('')

        logging.info('[InferenceDataset] load saved dataset')
        npz_files = sorted(glob(f'{self.dataset_config.cache_dir}/*/*npz'))
        total_npz_num = len(npz_files)
        self.dataset_len = total_npz_num * np.load(npz_files[0])['dG'].shape[0]
        if total_npz_num < trainer_num:
            tmp_data_list = InMemoryDataset(npz_data_files=npz_files)
            self.data_list = tmp_data_list[trainer_id::trainer_num]
        else:
            npz_files = npz_files[trainer_id::trainer_num]
            protein_array = []
            pocket_array = []
            ligand_array = []
            dG_array = []
            counter = 0
            for npz_file in npz_files:
                print('npz file : ', npz_file)
                protein_array.append(np.load(npz_file)['protein_name'])
                pocket_array.append(np.load(npz_file)['pocket_name'])
                ligand_array.append(np.load(npz_file)['ligand_name'])
                dG_array.append(np.load(npz_file)['dG'])
                counter += 1
                if counter > 5:
                    break
            protein_array = np.unique(np.concatenate(protein_array))
            pocket_array = np.unique(np.concatenate(pocket_array))
            ligand_array = np.unique(np.concatenate(ligand_array))
            dG_array = np.concatenate(dG_array)
            scale = len(npz_files) / 5.0
            if scale < 1:
                scale = 1
            self.protein_num = protein_array.shape[0] * scale
            self.pocket_num = pocket_array.shape[0]* scale
            self.ligand_num = ligand_array.shape[0]* scale
            self.dG = dG_array.mean()
            print(f'[InferenceDataset] Trianer {trainer_id} Dataset info : unique protein num : {self.protein_num} / unique pocket num : { self.pocket_num} / unique ligand num : {self.ligand_num}, sacle: {scale}, dG mean : {self.dG}, dataset len : {self.dataset_len}')
            self.data_list = InMemoryDataset(npz_data_files=npz_files)

    def _load_dataset(self, trainer_id, trainer_num):
        """
        TODO: add data_file
        suppose data is stored in a csv file:
        proatein_pdb_path, mol_file_path/smiles, affinity_pkd
        """
        with open(self.dataset_config.data_file) as f:
            lines = f.readlines()
            for line in lines:
                protein_path, ligand_path, 

        protein_names = [x.strip() for x in os.listdir(self.dataset_config.data_dir)] 
        protein_names = set(protein_names[trainer_id::trainer_num])
        # filter the complexes
        logging.info(f"Raw dataset has items: {len(protein_names)}.")

        protein_names = sorted(list(protein_names))
        data_list = []
        for name in protein_names:
            protein_path = f'{self.dataset_config.data_dir}/{name}.pdb'
            ligand_path = f'{self.dataset_config.data_dir}/{name}/{name}_ligand.mol2'
            if not exists(protein_path) or not exists(ligand_path):
                continue
            data = {
                'protein_name': name,
                'pocket_name': name + '-0',
                'ligand_name': "0",
                'protein_path': protein_path,
                'ligand_path': ligand_path,
                'dG': 0         # convert pIC50 to dG, we only count for atom pos for now
            }
            data_list.append(data)
        dataset = InMemoryDataset(data_list)
        return dataset

    def _read_label_file(self, file_path):
        """returns pKd"""
        key_label = []
        with open(file_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                colums = re.compile(r'\s+').split(line)
                # columns = line.split(',')
                key_label.append((colums[0], colums[3]))
        # df = pd.read_csv(file_path)
        
        complex_pKd_dict = {i[0]:float(i[1]) for i in key_label}
        return complex_pKd_dict

    def get_sample(self, raw_data):
        t = time.time()
        logging.info(f"StartTrans complex with name: {raw_data['pocket_name']} ")
        protein_mol = load_protein(raw_data['protein_path'])
        ligand_mol = load_ligand(raw_data['ligand_path'])
        if protein_mol is None:
            print('[NotExistProtein] ', raw_data['pocket_name'])
            return None
        if ligand_mol is None:
            print('[NotExistLigand] ', raw_data['pocket_name'])
            return None
        # TODO: extract pocket from box instead of ligand
        lig_mol, poc_mol, dis_mat = extract_pocket(ligand_mol, protein_mol)
        data_lig = new_mol_to_graph_data(lig_mol, if_fingerprint=True)
        data_poc = new_mol_to_graph_data(poc_mol, if_fingerprint=False)
        data_lig['atom_pos'] = lig_mol.GetConformers()[0].GetPositions().astype('float32')
        data_poc['atom_pos'] = poc_mol.GetConformers()[0].GetPositions().astype('float32')
        if self.data_config.gen_ETKDG_conformer:
            data_lig['ETKDG_atom_pos'] = generate_ETKDG_conformer(lig_mol)

        data = ligand_pocket_data_to_docking_data(data_lig, data_poc)
        data.update(raw_data)
        logging.info(f"Transforming complex with name: {raw_data['pocket_name']} "
                f"{raw_data['ligand_name']}, {time.time() - t}")
        return data
    
    def __len__(self):
        return len(self.data_list)
        #print('[PDBbind] dataset len : ', self.dataset_len)
        #return self.dataset_len
    
    def __getitem__(self, index):
        return self.data_list[index]