from copy import deepcopy
import hashlib
import logging
import time

import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.metrics import pairwise_distances
from scipy.spatial import distance_matrix

import paddle
import pgl
from pahelix.utils.compound_tools import new_mol_to_graph_data
from pahelix.utils.compound_tools import Compound3DKit 


class InterLiteGEMTransformFn(object):
    """Gen features for downstream model"""
    def __init__(self, model_config, encoder_config, is_inference=False):
        self.model_config = model_config
        self.encoder_config = encoder_config
        self.is_inference = is_inference

    def __call__(self, raw_data):
        """

        """
        lig_mol, poc_mol, dis_mat = self.extract_pocket(raw_data["ligand_mol"], raw_data['protein_mol'])
        logging.info(f"Transforming complex with name: {raw_data['complex_name']}")

        data_dict = {}
        # FIXME add more feature, check Kdeep
        data_lig = new_mol_to_graph_data(lig_mol, if_fingerprint=True)
        data_lig['pos'] = lig_mol.GetConformers()[0].GetPositions()

        data_poc = new_mol_to_graph_data(poc_mol, if_fingerprint=False)
        data_poc['pos'] = poc_mol.GetConformers()[0].GetPositions()
        # rename the key, which is used to distinguish the ligand and protein in the collate_func 
        keys = [k for k in data_poc]
        for key in keys:
            data_poc["protein_"+key] = data_poc.pop(key)

        data_dict.update(data_lig)
        data_dict.update(data_poc)
        data_dict['label'] = raw_data['pKd']
        data_dict['complex_name'] = raw_data['complex_name']
        return data_dict

    @staticmethod
    def extract_pocket(lig_mol, pro_mol, theta=8):
        # find distant protein atoms to be removed
        dis_matrix = distance_matrix(lig_mol.GetConformers()[0].GetPositions(), 
                                     pro_mol.GetConformers()[0].GetPositions())
        pro_list = dis_matrix.min(axis=0) > 8
        pro_set_toberemoved = np.where(pro_list)[0]
        
        # remove atoms
        mw = Chem.RWMol(pro_mol)
        mw.BeginBatchEdit()
        for i in pro_set_toberemoved[::-1]:     # from the end
            mw.RemoveAtom(int(i))
        mw.CommitBatchEdit()
        pro_mol = Chem.Mol(mw)

        poc_dis_matrix = dis_matrix[:, ~pro_list]
        return lig_mol, pro_mol, poc_dis_matrix


class InterLiteGEMCollateFn(object):
    """Gen features for downstream model"""
    def __init__(self, model_config, encoder_config, is_inference=False):
        self.model_config = model_config
        self.encoder_config = encoder_config
        self.is_inference = is_inference

        self.config = self.encoder_config.embedding_layer

    def __call__(self, raw_data):
        cov_graph_list = []
        noncov_graph_list = []
        labels = []
        complex_names = []
        for feature_dict in raw_data:
            new_graph = {}
            lig_anum = len(feature_dict['atomic_num'])
            pro_anum = len(feature_dict['protein_atomic_num'])
            total_anum = lig_anum + pro_anum
            node_types = list(zip(range(total_anum), ["lig"]*lig_anum + ['pro']*pro_anum))

            # be careful about the edge index of proteins, which should be added by the number of ligand atoms
            dis_matrix = distance_matrix(feature_dict['pos'], feature_dict['protein_pos'])
            dis_matrix = dis_matrix.astype(np.float32)
            node_idx = np.where(dis_matrix < self.encoder_config.non_cov_dis_cutoff)
            src_ls3 = np.concatenate([node_idx[0], node_idx[1] + lig_anum])
            dst_ls3 = np.concatenate([node_idx[1] + lig_anum, node_idx[0]])
            edges = {
                'cov': np.vstack([feature_dict['edges'], feature_dict['protein_edges'] + lig_anum]),
                'noncov': np.array(list(zip(src_ls3, dst_ls3)))
            }

            # concat features of ligands and proteins
            lig_nfeats = {key: feature_dict[key] for key in self.config.atom_names + self.config.atom_float_names}
            pro_nfeats = {key: feature_dict['protein_'+key] for key in self.config.atom_names + self.config.atom_float_names}
            all_nfeats = {key: np.concatenate([lig_nfeats[key], pro_nfeats[key]]) for key in lig_nfeats}

            lig_efeats = {key: feature_dict[key] for key in self.config.bond_names}
            pro_efeats = {key: feature_dict['protein_'+key] for key in self.config.bond_names}
            all_cov_efeats = {key: np.concatenate([lig_efeats[key], pro_efeats[key]]) for key in lig_efeats}
            all_efeats = {
                'cov': all_cov_efeats,
                'noncov': {"dis": np.repeat(dis_matrix[node_idx[0], node_idx[1]], 2)}    # distances between P and L
            }

            # graph = pgl.HeterGraph(
            #     edges=edges,
            #     node_types=node_types,
            #     node_feat=all_nfeats,
            #     edge_feat=all_efeats
            # )
            cov_g = pgl.Graph(
                num_nodes=total_anum,
                edges=edges['cov'],
                node_feat=all_nfeats,
                edge_feat=all_efeats['cov']
            )
            noncov_g = pgl.Graph(
                num_nodes=total_anum,
                edges=edges['noncov'],
                node_feat={'pseudo-feat': np.zeros(0)},    # pseudo-feat
                edge_feat=all_efeats['noncov']
            )

            cov_graph_list.append(cov_g)
            noncov_graph_list.append(noncov_g)
            labels.append(float(feature_dict['label']))
            complex_names.append(feature_dict['complex_name'])

        return {'cov_graph': cov_graph_list, 'non_cov_graph': noncov_graph_list,
                'labels': labels, 'complex_names': complex_names}
