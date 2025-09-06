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
config
"""
import numpy as np
import copy
import json
import ml_collections


def make_updated_config(base_config, updated_dict):
    """tbd"""
    if isinstance(updated_dict, str):
        # regard `updated_dict` as a json path
        updated_dict = json.load(open(updated_dict, 'r'))
    config = copy.deepcopy(base_config)
    config.update_from_flattened_dict(updated_dict)
    return config


PREDICTOR_CONFIG = ml_collections.ConfigDict({
    "data": {
        "gen_ETKDG_conformer": True,
        "max_pocket_atom_num": 300,
    },

    "model": {
        "encoder_type": "HelixDock",
        "diffusion_params":{
            "beta_1": 1e-4,
            "beta_T": 0.02,
            "T": 1000,
            "normal_mean": 10.0,
            "normal_std": 1.0,
            "label_mean": 0.0,
            "label_std": 1.0,
            "mean_type": "xstart",
            "in_use": False,
        },
        "heads": {
            "affinity_head": {
                "loss_scale": 1.0,
                "hidden_size": [128, 64],
                "output_size": 1,
                "loss_type": "l1loss",
                "label_mean": [0.0],
                "label_std": [1.0],
                "in_use": False,
            },
            "confidence_head": {
                "loss_scale": 1.0,
                "hidden_size": [128, 64],
                "in_use": False,
            },
            "affinity_class_head": {
                "loss_scale": 1.0,
                "hidden_size": [128, 64],
                "output_size": 1,
                "loss_type": "bceloss",
                "label_mean": [0.0],
                "label_std": [1.0],
                "in_use": False,
            },
            "multilayer_affinity_head": {
                "loss_scale": 1.0,
                "hidden_size": [128, 64],
                "output_size": 1,
                "loss_type": "l1loss",
                "label_mean": [0.0],
                "label_std": [1.0],
                "in_use": False,
            },
            "ligand_atom_pos_head": {
                "loss_scale": 1.0,
                "loss_type": "l1loss",
                "label_mean": [0.0],
                "label_std": [1.0],
                "in_use": True,
            },
            "affinity_biclass_head": {
                "output_size": 1,
                "hidden_size": [128, 64],
                "loss_scale": 1.0,
                "loss_type": "bce_loss",
                "in_use": False,
            },
            "affinity_pairclass_head": {
                "output_size": 1,
                "hidden_size": [128, 64],
                "loss_scale": 1.0,
                "loss_type": "bce_loss",
                "in_use": False,
            },
            "contactmap_head":{
                "contactmap_class": 30,
                "hidden_size": [128, 64],
                "loss_scale": 1.0,
                "in_use": False,
            },
        },
    },
})


InterLiteGEM_MODEL_CONFIG = ml_collections.ConfigDict({
    "num_layers": 8,
    "triangle_layers": 1,
    "dropout": 0.1,
    "add_virtual_node": False,
    "add_virtual_edge": False,
    "graph_feat_size": 128,
    "graph_feat_size": 128,
    "node_feat_size": 128,
    "edge_feat_size": 128,
    "node_channel": 128,
    "non_cov_dis_cutoff": 6,
    "mlp_layers": 2,

    "embedding_layer": {
        "atom_names": ["atomic_num", "formal_charge", "degree", 
            "chiral_tag", "total_numHs", "is_aromatic", 
            "hybridization"],
        "atom_float_names": ["van_der_waals_radis", "partial_charge"],  # remove 'mass'
        "bond_names": ["bond_dir", "bond_type", "is_in_ring", "is_conjugated", "bond_stereo"],
        "bond_float_names": ["dis"],
        "triple_names": ["hop_num_ij", "hop_num_ik", "hop_num_jk"],
        "triple_float_names": ["angle_i", "angle_j", "angle_k"],

        "rbf_params": {
            "bond_length": [0, 5, 0.1, 10.0],
            "angle_i": [0, np.pi, 0.1, 10.0],
            "angle_j": [0, np.pi, 0.1, 10.0],
            "angle_k": [0, np.pi, 0.1, 10.0],
        },
    },
})


HelixDock_MODEL_CONFIG = ml_collections.ConfigDict({
    "max_recycle_num": 0,

    "atom_channel": 128,
    "edge_channel": 128,

    "embedding_layer": {
        "ligand_atom_names": ["atomic_num", "formal_charge", "degree", 
            "chiral_tag", "total_numHs", "is_aromatic", 
            "hybridization"],
        "ligand_atom_float_names": ["van_der_waals_radis", "partial_charge"],  # remove 'mass'
        "ligand_bond_names": ["bond_dir", "bond_type", "is_in_ring", "is_conjugated", "bond_stereo"],
        "ligand_bond_float_names": ["bond_length"],
        
        "protein_atom_names": ["atomic_num", "formal_charge", "degree", 
            "chiral_tag", "total_numHs", "is_aromatic", 
            "hybridization"],
        "protein_atom_float_names": ["van_der_waals_radis", "partial_charge"],  # remove 'mass'
        "protein_bond_names": ["bond_dir", "bond_type", "is_in_ring", "is_conjugated", "bond_stereo"],
        "protein_bond_float_names": ["bond_length"],

        "rbf_params": {
            'van_der_waals_radis': {
                "start": 1,
                "end": 3,
                "stride": 0.2,
                "gamma": 10.0,
            },
            'partial_charge': {
                "start": -1,
                "end": 4,
                "stride": 0.25,
                "gamma": 10.0,
            },
            'bond_length': {
                "start": 0,
                "end": 5,
                "stride": 0.1,
                "gamma": 10.0,
            }
        }
    },

    "init_ligand_atom_pos_name": "zero",
    "init_dropout_rate": 0.1,

    "helixdock_block_num": 8,
    "helixdock_block": {
        "dropout_rate": 0.1,
        "ligand_cov_gnn": {
            "graph_feat_size": 128,
            "mlp_layers": 2,
            "norm": "layer",
        },
        "protein_cov_gnn": {
            "graph_feat_size": 128,
            "mlp_layers": 2,
            "norm": "layer",
        },
        "ligand_attn": {
            "num_head": 8,
            "dropout_rate": 0.1,
        },
        "ligand_ffn": {
            "hidden_factor": 4,
            "dropout_rate": 0.2,
        },
        "protein_attn": {
            "num_head": 8,
            "dropout_rate": 0.1,
        },
        "protein_ffn": {
            "hidden_factor": 4,
            "dropout_rate": 0.2,
        },
        "rbf_params": {
            'ligand_dist': {
                "start": 0,
                "end": 30,
                "stride": 1,
                "gamma": 10.0,
            },
            'protein_dist': {
                "start": 0,
                "end": 30,
                "stride": 1,
                "gamma": 10.0,
            },
        }
    },
    "helixdock_e3_block_num": 6,
    "helixdock_e3_block": {
        "dropout_rate": 0.1,
        "ligand_cov_gnn": {
            "graph_feat_size": 128,
            "mlp_layers": 2,
            "norm": "layer",
        },
        "protein_cov_gnn": {
            "graph_feat_size": 128,
            "mlp_layers": 2,
            "norm": "layer",
        },
        "ligand_attn": {
            "num_head": 8,
            "dropout_rate": 0.1,
        },
        "ligand_ffn": {
            "hidden_factor": 4,
            "dropout_rate": 0.2,
        },
        "protein_attn": {
            "num_head": 8,
            "dropout_rate": 0.1,
        },
        "protein_ffn": {
            "hidden_factor": 4,
            "dropout_rate": 0.2,
        },
        "e3_attn": {
            "num_head": 8,
            "dropout_rate": 0.1,
        },
        "rbf_params": {
            'ligand_dist': {
                "start": 0,
                "end": 30,
                "stride": 1,
                "gamma": 10.0,
            },
            'protein_dist': {
                "start": 0,
                "end": 30,
                "stride": 1,
                "gamma": 10.0,
            },
        },
        "diffusion_params": {
            "beta_1": 1e-4,
            "beta_T": 0.02,
            "T": 1000,
        },
    },
    "helixdock_seq_block_num":0,
    "helixdock_seq_block": {
        "dropout_rate": 0.1,
        "ligand_cov_gnn": {
            "graph_feat_size": 128,
            "mlp_layers": 2,
            "norm": "layer",
        },
        "protein_cov_gnn": {
            "graph_feat_size": 128,
            "mlp_layers": 2,
            "norm": "layer",
        },
        "ligand_attn": {
            "num_head": 8,
            "dropout_rate": 0.1,
        },
        "ligand_ffn": {
            "hidden_factor": 4,
            "dropout_rate": 0.2,
        },
        "protein_attn": {
            "num_head": 8,
            "dropout_rate": 0.1,
        },
        "protein_ffn": {
            "hidden_factor": 4,
            "dropout_rate": 0.2,
        },
        "rbf_params": {
            'ligand_dist': {
                "start": 0,
                "end": 30,
                "stride": 1,
                "gamma": 10.0,
            },
            'protein_dist': {
                "start": 0,
                "end": 30,
                "stride": 1,
                "gamma": 10.0,
            },
        }
    },
    "helixdock_ipa_block_num": 0,
    "helixdock_ipa_block": {
        "ipa": {
            "num_head": 8,
            "num_point_qk": 4,
            'num_point_v': 8,
            'num_scalar_qk': 16,
            'num_scalar_v': 16,
            "final_act_ffn":{
                "hidden_factor": 4,
                "dropout_rate": 0.2,
            },
        },
        "dropout_rate": 0.1,      
        "ligand_cov_gnn": {
            "graph_feat_size": 128,
            "mlp_layers": 2,
            "norm": "layer",
        },
        "protein_cov_gnn": {
            "graph_feat_size": 128,
            "mlp_layers": 2,
            "norm": "layer",
        },
        "ligand_attn": {
            "num_head": 8,
            "dropout_rate": 0.1,
        },
        "ligand_ffn": {
            "hidden_factor": 4,
            "dropout_rate": 0.2,
        },
        "protein_attn": {
            "num_head": 8,
            "dropout_rate": 0.1,
        },
        "protein_ffn": {
            "hidden_factor": 4,
            "dropout_rate": 0.2,
        },
        "rbf_params": {
            'ligand_dist': {
                "start": 0,
                "end": 30,
                "stride": 1,
                "gamma": 10.0,
            },
            'protein_dist': {
                "start": 0,
                "end": 30,
                "stride": 1,
                "gamma": 10.0,
            },
        },
    },
})

