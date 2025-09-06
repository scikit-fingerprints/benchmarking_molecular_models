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
# Unless required by applicable law or agreed to in writing, sosftware
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, eitdher express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Featurizers
"""

import os
import time
import numpy as np
import gzip
import pickle
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import SanitizeFlags

import pgl

from scipy.spatial import distance_matrix

from pahelix.utils.compound_tools import Compound3DKit
from pahelix.utils.compound_tools import mol_to_transformer_data, mol_to_trans_data_w_meta_path, mol_to_trans_data_w_pair_dist
from pahelix.utils.compound_tools import mol_to_trans_data_w_rdkit3d
from pahelix.utils.compound_tools import new_mol_to_graph_data
from scipy.spatial.transform import Rotation as R

from .utils import sequence_pad, sequence_mask, edge_to_pair, pair_pad, sequence_pad_2d


class HelixDockCollateFn(object):
    """HelixDockCollateFn"""
    def __init__(self, model_config, encoder_config, is_inference=False):
        self.model_config = model_config
        self.encoder_config = encoder_config
        self.is_inference = is_inference

        self.embed_config = self.encoder_config.embedding_layer
    def _get_ligand_atom_nums(self, data_list):
        return [len(x['ligand_' + self.embed_config.ligand_atom_names[0]]) 
                for x in data_list]

    def _get_ligand_chiral_center_nums(self, data_list):
        center_num_list = []
        for x in data_list:
            center_num = len(x['ligand_chiral_center'])
            if center_num == 1:
                if np.sum(x['ligand_chiral_label']) <= 0:
                    # for the ligand without chiral cetner, we will mark its chiral center pos as [0,0,0]
                    center_num = 0
            center_num_list.append(center_num)
        return center_num_list

    def _get_protein_atom_nums(self, data_list):
        return [len(x['protein_' + self.embed_config.protein_atom_names[0]]) 
            for x in data_list]

    def _batch_sequence_pad(self, data_list, names):
        """
        :return:
            batch: {
                name: (B, max_len, *)
            }
        """
        batch = {}
        for name in names:
            try:
                if name == 'ligand_chiral_neighbor_center':
                    # for some of the chiral centers that not containing H as neighbor
                    tmp_list = [x[name][:, :3] for x in data_list]
                    batch[name] = sequence_pad(tmp_list)
                else:
                    batch[name] = sequence_pad([x[name] for x in data_list])
            except:
                print(f'[DEBUG] failed on{name} : ', [x[name].shape for x in data_list]) 
        return batch

    def _batch_sequence_concat(self, data_list, names):
        """
        :return:
            batch: {
                name: (B_L, *)
            }
        """
        batch = {}
        for name in names:
            batch[name] = np.concatenate([x[name] for x in data_list], 0)
        return batch
            
    def aggregate_features(self, data_list):
        atom_names = list(map(lambda x: 'ligand_' + x, self.embed_config.ligand_atom_names))
        atom_names += list(map(lambda x: 'ligand_' + x, self.embed_config.ligand_atom_float_names))
        atom_names += list(map(lambda x: 'protein_' + x, self.embed_config.protein_atom_names))
        atom_names += list(map(lambda x: 'protein_' + x, self.embed_config.protein_atom_float_names))
        atom_names += ['ligand_zero_atom_pos',
                'ligand_atom_pos', 'protein_atom_pos']

        bond_names = list(map(lambda x: 'ligand_' + x, self.embed_config.ligand_bond_names))
        bond_names += list(map(lambda x: 'ligand_' + x, self.embed_config.ligand_bond_float_names))
        bond_names += list(map(lambda x: 'protein_' + x, self.embed_config.protein_bond_names))
        bond_names += list(map(lambda x: 'protein_' + x, self.embed_config.protein_bond_float_names))
        
        batch = {}
        batch.update(self._batch_sequence_pad(data_list, atom_names))
        batch.update(self._batch_sequence_concat(data_list, bond_names))

        ## extra
        ligand_atom_nums = self._get_ligand_atom_nums(data_list)
        protein_atom_nums = self._get_protein_atom_nums(data_list)
        batch['ligand_atom_mask'] = sequence_mask(ligand_atom_nums).astype('float32')
        batch['protein_atom_mask'] = sequence_mask(protein_atom_nums).astype('float32')
        batch['ligand_cov_graph'] = pgl.Graph.batch([
            pgl.Graph(
                num_nodes=np.max(ligand_atom_nums),     # we pad each graph to max_atom_num
                edges=data['ligand_edges'],
            ) for data in data_list
        ])
        batch['protein_cov_graph'] = pgl.Graph.batch([
            pgl.Graph(
                num_nodes=np.max(protein_atom_nums),
                edges=data['protein_edges'],
            ) for data in data_list
        ])
        return batch
    
    def __call__(self, data_list):
        """
        # valid_mask is for neg sampling, 
        # 1 stands for original pdbbind data, count affinity and pos loss, 
        # 0 stands for neg sampled data, only count for affinity biclass loss
        """
        batch = self.aggregate_features(data_list)

        # graph-level names
        int_names = ['is_active', 'bi_class_dG']
        float_names = ['dG', 'vina_dG', 'valid_mask']
        str_names = ['protein_name', 'pocket_name', 'ligand_name', 'protein_path', 'ligand_path']
        for name in int_names:
            if name in data_list[0]:
                batch[name] = np.array([d[name] for d in data_list], 'int64')
        for name in float_names:
            if name in data_list[0]:
                batch[name] = np.array([d[name] for d in data_list], 'float32')
        for name in str_names:
            if name in data_list[0]:
                batch[name] = [d[name] for d in data_list]
        return batch

class PostRefineCollateFn(HelixDockCollateFn):
    def __init__(self, model_config, encoder_config, is_inference=False):
        super(PostRefineCollateFn, self).__init__(model_config, encoder_config)

    def get_tree_frame_idx(self, frame_idx, torsion_bond_index, frame_heavy_atoms_matrix):
        rotorY_index = int(torsion_bond_index[frame_idx][1])
        cur_atoms_index = np.where(frame_heavy_atoms_matrix[rotorY_index]== 1)[0].tolist()
        other_ids = []
        def get_other_idx(cur_idx):
            nonlocal other_ids
            for id in cur_idx:
                for u, v in torsion_bond_index:
                    u, v = int(u), int(v)
                    if id == u:
                        cr_list = np.where(frame_heavy_atoms_matrix[v] == 1)[0].tolist()
                        other_ids.extend(cr_list)
                        get_other_idx(cr_list)

        get_other_idx(cur_atoms_index)
        f_atoms_index = sorted(list(set(other_ids) | set(cur_atoms_index)))
        return f_atoms_index

    def aggregate_features(self, data_list):
        atom_names = list(map(lambda x: 'ligand_' + x, self.embed_config.ligand_atom_names))
        atom_names += list(map(lambda x: 'ligand_' + x, self.embed_config.ligand_atom_float_names))
        atom_names += list(map(lambda x: 'protein_' + x, self.embed_config.protein_atom_names))
        atom_names += list(map(lambda x: 'protein_' + x, self.embed_config.protein_atom_float_names))
        atom_names += ['ligand_zero_atom_pos',
                'ligand_atom_pos', 'protein_atom_pos']
        
        bond_names = list(map(lambda x: 'ligand_' + x, self.embed_config.ligand_bond_names))
        bond_names += list(map(lambda x: 'ligand_' + x, self.embed_config.ligand_bond_float_names))
        bond_names += list(map(lambda x: 'protein_' + x, self.embed_config.protein_bond_names))
        bond_names += list(map(lambda x: 'protein_' + x, self.embed_config.protein_bond_float_names))
        
        batch = {}
        batch.update(self._batch_sequence_pad(data_list, atom_names))
        batch.update(self._batch_sequence_concat(data_list, bond_names))

        ## extra
        ligand_atom_nums = self._get_ligand_atom_nums(data_list)
        protein_atom_nums = self._get_protein_atom_nums(data_list)
        batch['ligand_atom_mask'] = sequence_mask(ligand_atom_nums).astype('float32')
        batch['protein_atom_mask'] = sequence_mask(protein_atom_nums).astype('float32')
        batch['ligand_cov_graph'] = pgl.Graph.batch([
            pgl.Graph(
                num_nodes=np.max(ligand_atom_nums),     # we pad each graph to max_atom_num
                edges=data['ligand_edges'],
            ) for data in data_list
        ])
        batch['protein_cov_graph'] = pgl.Graph.batch([
            pgl.Graph(
                num_nodes=np.max(protein_atom_nums),
                edges=data['protein_edges'],
            ) for data in data_list
        ])

        max_k, max_atoms = 0, 0
        for feature_dict in data_list:
            max_k = max(max_k, feature_dict['number_of_frames'])
            max_atoms = max(max_atoms, feature_dict['number_of_heavy_atoms'])

        number_of_frames_list, number_of_heavy_atoms_list, ligand_6k_list, ligand_init_xyz_list, \
        ligand_center_list, torsion_bond_index_list, frame_heavy_dict_list, frame_heavy_atoms_matrix_list = [], [], [], [], [], [], [], []
        
        for feature_dict in data_list:

            number_of_frames_list.append(feature_dict['number_of_frames'])  # k
            number_of_heavy_atoms_list.append(feature_dict['number_of_heavy_atoms'])  # atom_nums
            ligand_6k_list.append(
                np.concatenate([feature_dict['6k'], np.zeros(shape=(1, max_k - feature_dict['number_of_frames']))], axis=-1))
            ligand_init_xyz_list.append(np.concatenate(
                [feature_dict['init_lig_heavy_atoms_xyz'], np.zeros(shape=(1, max_atoms - feature_dict['number_of_heavy_atoms'], 3))],
                axis=1))
            ligand_center_list.append(feature_dict['ligand_center'])

            if feature_dict['number_of_frames'] != 0:  
                torsion_bond_index_list.append(np.pad(feature_dict['torsion_bond_index'],
                                                        ((0, max_k - feature_dict['number_of_frames']),(0,0)),'constant', constant_values=(0,0)))
            else:
                torsion_bond_index_list.append(np.zeros(shape=(max_k, 2)))

            frame_heavy_atoms_matrix_list.append(np.pad(feature_dict['frame_heavy_atoms_matrix'],
                                                        ((0, max_atoms - feature_dict['number_of_heavy_atoms']),
                                                        (0, max_atoms - feature_dict['number_of_heavy_atoms'])),
                                                        'constant', constant_values=(0, 0)))

            tmp_list = []
            for frame_idx in range(feature_dict['number_of_frames']):
                tree_frame_idxs = self.get_tree_frame_idx(frame_idx, feature_dict['torsion_bond_index'], feature_dict['frame_heavy_atoms_matrix'])
                tmp_list.append(np.pad(np.array(tree_frame_idxs),(0,max_atoms - len(tree_frame_idxs)), 'constant', constant_values=(0, -1)))

            if len(tmp_list) != 0: 
                frame_heavy_dict_list.append(np.pad(np.array(tmp_list), ((0, max_k-feature_dict['number_of_frames']), (0, 0)),'constant', constant_values=(0, -1)).astype(np.int32))
            else:
                frame_heavy_dict_list.append((np.zeros(shape=(max_k, max_atoms))-1).astype(np.int32))

        data_dict = {'number_of_frames': number_of_frames_list,
                'number_of_heavy_atoms': number_of_heavy_atoms_list,
                '6k': ligand_6k_list,
                'init_xyz': ligand_init_xyz_list,
                'center_xyz': ligand_center_list,
                'torsion_bond': torsion_bond_index_list,
                'frame_heavy_atoms_matrix': frame_heavy_atoms_matrix_list,
                'frame_heavy_dict_list': frame_heavy_dict_list, }

        batch.update(data_dict)
        return batch
    
    def __call__(self, data_list):
        # {feature_key: list}
        batch = self.aggregate_features(data_list)

        # graph-level names
        int_names = ['is_active', 'bi_class_dG']
        print('data_list keys : ', data_list[0].keys() )
        if 'interactions_HBonds' in data_list[0]:
            print('appending interaction feats ! ')
            int_names += ['interactions_HBonds', 'interactions_halogenBonds', 'interactions_hydro_contact', 'interactions_salt_bridges', 'interactions_acceptor_metal']
            int_names += ['noncov_nodes']
        float_names = ['dG', 'vina_dG', 'pkd']
        str_names = ['protein_name', 'pocket_name', 'ligand_name', 'protein_path', 'ligand_path', 'docking_path']
        for name in int_names:
            if name in data_list[0]:
                batch[name] = np.array([d[name] for d in data_list], 'int64')
        for name in float_names:
            if name in data_list[0]:
                batch[name] = np.array([d[name] for d in data_list], 'float32')
        for name in str_names:
            if name in data_list[0]:
                batch[name] = [d[name] for d in data_list]

        # 6k-level features, to ndarray shape.
        #  'frame_heavy_dict_list' may be not necessary for converting.
        for k in ['number_of_frames', 'number_of_heavy_atoms', 'torsion_bond', 'frame_heavy_dict_list']:
            batch[k] = np.array(batch[k]).astype('int64')
        for k in  ['6k', 'init_xyz', 'center_xyz', 'frame_heavy_atoms_matrix']:
            batch[k] = np.array(batch[k]).astype('float32')
        print('handle batch done')
        return batch
