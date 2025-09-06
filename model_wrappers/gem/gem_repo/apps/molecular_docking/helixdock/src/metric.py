# !/usr/bin/env python3
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
metric
"""

import sys
import os
from os.path import exists, dirname, basename, join
from unittest import result
import numpy as np
import logging

from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

import paddle
import paddle.distributed as dist

from .paddle_utils import dist_mean, dist_sum, dist_length, dist_all_gather_object
from .utils import tree_map, tree_filter, tree_flatten
from .mol_utils import load_ligand, save_atom_pos_list_to_sdf, get_obrms_rmsd
from rdkit.ML.Scoring import Scoring
import pandas as pd
from rdkit.Chem.Draw import rdMolDraw2D
import copy

import pdb

def gather_all(array_to_gather, name):
    """tbd"""
    print(f'{name} shape ', array_to_gather.shape)
    np.savez("gather_dir/rank_{}_{}.npz".format(name, dist.get_rank()), attr=array_to_gather)
    dist.barrier()
    pock_name_list = []
    for i in range(dist.get_world_size()):
        try:
            pock_name_list.append(np.load(f"gather_dir/rank_{name}_{i}.npz")['attr'])
        except:
            print(f'[Gather_failed] rank {i}, {name}')
    all_pocket_names = np.concatenate(pock_name_list)
    return all_pocket_names

def calc_ligand_rmsd(pred_pos, label_pos, mask):
    """
    pred_pos: (N, 3)
    label_pos: (N, 3)
    mask: (N), 1 for valid and 0 for invalid
    """
    dist2 = np.sum((pred_pos - label_pos) ** 2, 1) * mask
    rmsd = np.sqrt(dist2.sum() / mask.sum())
    return rmsd


# added3
def distance_matrix(pos1, pos2):
    """
    pos1: (N1, 3)
    pos2: (N2, 3)
    return: 
        dist: (N1, N2)
    """
    assert len(pos1.shape) == 2
    pos1 = pos1[:, np.newaxis, :]
    pos2 = pos2[np.newaxis, :, :]
    dist = np.sqrt(np.sum((pos1 - pos2) ** 2, -1) + 1e-5) # (N1, N2)
    return dist


class Metric(object):
    """tbd"""
    def __init__(self, save_output=False, output_dir=None, output_num=100, save_ligand_rmsd=True):
        self.save_output = save_output
        self.save_ligand_rmsd = save_ligand_rmsd 
        print('save_ligand_rmsd: ', save_ligand_rmsd)
        self.output_dir = output_dir
        print('output_dir: ', output_dir)
        os.makedirs(output_dir, exist_ok=True)
        if self.save_output:
            self.output_num = output_num
            self.output_count = 0
        
        self.pocket_names = []
        self.dG_preds = []
        self.dG_labels = []
        self.ligand_rmsds = []
        self.ligand_obrms_rmsds = []
        self.batch_counter = 0
        self.save_prmsd = False
        self.prmsds = []
        self.ligand_atom_prmsd = []
        self.ligand_atom_prmsd_label = []
        self.ligand_atom_prmsd_pr = []
        self.atom_auc_fail_count = 0
        self.atom_auc_fail_count1 = 0

    def add(self, batch, results):
        """tbd"""
        self.pocket_names += batch['pocket_name']
        ligand_pred_pos = results['ligand_atom_pos_head']['final_pred_pos'].numpy() # (B, N, 3)
        ligand_atom_pos = batch['ligand_atom_pos'].numpy()    # (B, N, 3)
        ligand_atom_mask = batch['ligand_atom_mask'].numpy()  # (B, N)
        complex_name_list = [f'{n1}-{n2}' for n1, n2 
                in zip(batch['pocket_name'], batch['ligand_name'])]
        ligand_path_list = batch['ligand_path']
        ligand_path_list = [i.replace('alphafold', 'docking') for i in ligand_path_list]
        ligand_all_pred_pos = [x.numpy() 
                for x in results['ligand_atom_pos_head']['all_pred_pos']]  # [(B, N, 3)]
        rsmd_file = open(f'{self.output_dir}/rmsd_result.csv', 'a')
        for i in range(len(ligand_atom_pos)):
            rmsd = calc_ligand_rmsd(ligand_pred_pos[i], ligand_atom_pos[i], ligand_atom_mask[i])
            atom_mask = ligand_atom_mask[i] == 1
            atom_pos_list = [x[i][atom_mask] for x in ligand_all_pred_pos]
            atom_pos_list = [atom_pos_list[-1]]
            sdf_file = f'{self.output_dir}/{complex_name_list[i]}.sdf'
            mol = load_ligand(ligand_path_list[i])
            save_success = save_atom_pos_list_to_sdf(mol, atom_pos_list, sdf_file)
            if not save_success:
                print('cal obrms failed on ', complex_name_list[i], 'it may casued by cannot kekulize mol')
            else:
                if '_obabel' in ligand_path_list[i]:
                    ligand_path_list[i] = ligand_path_list[i].split('_obabel')[0]
                obrms_rmsd = get_obrms_rmsd(ligand_path_list[i], sdf_file)
                self.ligand_rmsds.append(rmsd)
                self.ligand_obrms_rmsds.append(obrms_rmsd)
                rsmd_file.write('%s, %.3f, %.3f\n' % (complex_name_list[i], rmsd, obrms_rmsd))
                
        rsmd_file.close()

    def get_result(self, distributed=False):
        """tbd"""
        pocket_names = np.array(self.pocket_names)
        dG_preds = np.array(self.dG_preds)
        dG_labels = np.array(self.dG_labels)
        ligand_rmsds = np.array(self.ligand_rmsds)
        # added3
        ligand_obrms_rmsds = np.array(self.ligand_obrms_rmsds)

        if distributed:
            try:
                pocket_names = np.concatenate(dist_all_gather_object(pocket_names), 0)
                dG_preds = np.concatenate(dist_all_gather_object(dG_preds), 0)
                dG_labels = np.concatenate(dist_all_gather_object(dG_labels), 0)
                ligand_rmsds = np.concatenate(dist_all_gather_object(ligand_rmsds), 0)
                ligand_obrms_rmsds = np.concatenate(dist_all_gather_object(ligand_obrms_rmsds), 0)
            except:
                pocket_names = gather_all(pocket_names, 'pocket_names')
                dG_preds = gather_all(dG_preds, 'dG_preds')
                dG_labels = gather_all(dG_labels, 'dG_labels')
                ligand_rmsds = gather_all(ligand_rmsds, 'ligand_rmsds')
                ligand_obrms_rmsds = gather_all(ligand_obrms_rmsds, 'ligand_obrms_rmsds')

        if self.save_ligand_rmsd:
            with open(f'{self.output_dir}/ligand_rmsd.csv', 'w') as fcsv:
                fcsv.write('name,rmsd,obrms_rmsd\n')
                for pock_name, rmsd, obrms_rmsd in zip(pocket_names, ligand_rmsds, ligand_obrms_rmsds):
                    fcsv.write(f'{pock_name},{rmsd},{obrms_rmsd}\n')
 
        results = {
            "sample_num": len(ligand_obrms_rmsds),
            "ligand_obrms_rmsds": np.mean(ligand_obrms_rmsds),
            "ligand_obrms_rmsds_lt_2A": np.mean(ligand_obrms_rmsds < 2),
            "ligand_obrms_rmsds_lt_5A": np.mean(ligand_obrms_rmsds < 5),
        }
        return results


class ResultsCollect(object):
    """ResultsCollect"""
    def __init__(self):
        self.dict_to_mean_list = []
        self.dict_to_sum_list = []

    def add(self, batch, results, dict_to_mean=None, dict_to_sum=None):
        """
        batch, results: 
        dict_to_mean: {key: float, ...}
        dict_to_sum: {key: float, ...}
        """
        if dict_to_mean is None:
            dict_to_mean = {}
        if dict_to_sum is None:
            dict_to_sum = {}

        loss_dict = self._extract_loss_dict(results)
        dict_to_mean.update(loss_dict)

        if len(dict_to_mean) > 0:
            self.dict_to_mean_list.append(dict_to_mean)
        if len(dict_to_sum) > 0:
            self.dict_to_sum_list.append(dict_to_sum)

    def get_result(self, distributed=False):
        """tbd"""
        result = {}
        result.update(self._get_mean_result(distributed))
        result.update(self._get_sum_result(distributed))
        return result

    def _get_mean_result(self, distributed=False):
        result = {}
        if len(self.dict_to_mean_list) == 0:
            return result
        keys = list(self.dict_to_mean_list[0].keys())
        for k in keys:
            result[k] = dist_mean(
                    [d[k] for d in self.dict_to_mean_list], 
                    distributed=distributed)
        return result
    
    def _get_sum_result(self, distributed=False):
        result = {}
        if len(self.dict_to_sum_list) == 0:
            return result
        keys = list(self.dict_to_sum_list[0].keys())
        for k in keys:
            result[k] = dist_sum(
                    [d[k] for d in self.dict_to_sum_list], 
                    distributed=distributed)
        return result

    def _extract_loss_dict(self, results):
        """extract value with 'loss' in key"""
        res = tree_flatten(results)
        res = tree_filter(lambda k: 'loss' in k, None, res)
        res = tree_map(lambda x: x.numpy().mean(), res)
        return res
