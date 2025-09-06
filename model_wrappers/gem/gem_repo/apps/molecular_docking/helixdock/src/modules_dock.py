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
Modules for HelixDock
"""

import numpy as np
import logging

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from pgl.nn import GraphPool

from .basic_block import IntEmbedding, RBFEmbedding
from .paddle_utils import recompute_wrapper

from pahelix.utils.compound_tools import CompoundKit
from pahelix.networks.gnn_block import MeanPool, GraphNorm
from pahelix.networks.basic_block import MLP
from pahelix.model_zoo.light_gem_model import LiteGEMConv, norm_layer

from .diffusion import TimeEmbedding, Swish


def distance_matrix(pos1, pos2):
    """
    pos1: (B, N1, 3)
    pos2: (B, N2, 3)
    return: 
        dist: (B, N1, N2)
    """
    assert len(pos1.shape) == 3
    pos1 = pos1.unsqueeze([2])
    pos2 = pos2.unsqueeze([1])
    dist = paddle.sqrt(paddle.sum(paddle.square(pos1 - pos2), -1) + 1e-5) # (B, N1, N2)
    return dist


def relative_position(pos1, pos2):
    """
    pos1: (B, N1, 3)
    pos2: (B, N2, 3)
    return:
        rel_pos: (B, N1, N2)
    """
    assert len(pos1.shape) == 3
    pos1 = pos1.unsqueeze([2])
    pos2 = pos2.unsqueeze([1])
    rel_pos = pos1 - pos2   # (B, N1, N2, 3)
    return rel_pos


class EmbeddingLayer(nn.Layer):
    """
    EmbeddingLayer
    """
    def __init__(self, model_config, global_config):
        super(EmbeddingLayer, self).__init__()
        self.model_config = model_config

        atom_channel = global_config.atom_channel
        edge_channel = global_config.edge_channel
        
        embed_params = self._get_embed_params()
        rbf_params = self.model_config.rbf_params

        # ligand
        self.ligand_atom_embed = IntEmbedding(
                self.model_config.ligand_atom_names, atom_channel, embed_params)
        self.ligand_atom_float_rbf = RBFEmbedding(
                self.model_config.ligand_atom_float_names, atom_channel, rbf_params)
        self.ligand_bond_embed = IntEmbedding(
                self.model_config.ligand_bond_names, edge_channel, embed_params)
        self.ligand_bond_float_rbf = RBFEmbedding(
                self.model_config.ligand_bond_float_names, edge_channel, rbf_params)
        
        # protein
        self.protein_atom_embed = IntEmbedding(
                self.model_config.protein_atom_names, atom_channel, embed_params)
        self.protein_atom_float_rbf = RBFEmbedding(
                self.model_config.protein_atom_float_names, atom_channel, rbf_params)
        self.protein_bond_embed = IntEmbedding(
                self.model_config.protein_bond_names, edge_channel, embed_params)
        self.protein_bond_float_rbf = RBFEmbedding(
                self.model_config.protein_bond_float_names, edge_channel, rbf_params)
    
    def _get_embed_params(self):
        embed_params = {k: {'vocab_size': len(v) + 5} for k, v in CompoundKit.atom_vocab_dict.items()}
        embed_params.update({k: {'vocab_size': len(v) + 5} for k, v in CompoundKit.bond_vocab_dict.items()})
        return embed_params
        
    def forward(self, batch):
        """
        tbd
        """
        ## ligand
        ligand_feat_names = self.model_config.ligand_atom_names \
                + self.model_config.ligand_atom_float_names \
                + self.model_config.ligand_bond_names \
                + self.model_config.ligand_bond_float_names
        ligand_feats = {name: batch[f'ligand_{name}'] for name in ligand_feat_names}  # rename "ligand_xx" to "xx"
        ligand_atom_acts = self.ligand_atom_embed(ligand_feats)     # (B, N, d)
        ligand_atom_acts += self.ligand_atom_float_rbf(ligand_feats)
        ligand_bond_acts = self.ligand_bond_embed(ligand_feats)     # (B_E, d)
        ligand_bond_acts += self.ligand_bond_float_rbf(ligand_feats)

        ## protein
        protein_feat_names = self.model_config.protein_atom_names \
                + self.model_config.protein_atom_float_names \
                + self.model_config.protein_bond_names \
                + self.model_config.protein_bond_float_names
        protein_feats = {name: batch[f'protein_{name}'] for name in protein_feat_names}
        protein_atom_acts = self.protein_atom_embed(protein_feats)     # (B, N', d)
        protein_atom_acts += self.protein_atom_float_rbf(protein_feats)
        protein_bond_acts = self.protein_bond_embed(protein_feats)     # (B_E', d)
        protein_bond_acts += self.protein_bond_float_rbf(protein_feats)
        results = {
        'ligand_atom_acts': ligand_atom_acts,
        'ligand_bond_acts': ligand_bond_acts,
        'protein_atom_acts': protein_atom_acts,
        'protein_bond_acts': protein_bond_acts,
        }
        return results


class CrossAttention(nn.Layer):
    """CrossAttention"""
    def __init__(self, model_config, global_config):
        super().__init__()
        self.model_config = model_config

        atom_channel = global_config.atom_channel

        self.num_head = model_config.num_head
        self.head_dim = atom_channel // self.num_head

        self.q_ln = nn.LayerNorm(atom_channel)
        self.k_ln = nn.LayerNorm(atom_channel)

        self.q_proj = nn.Linear(atom_channel, atom_channel)
        self.k_proj = nn.Linear(atom_channel, atom_channel)
        self.v_proj = nn.Linear(atom_channel, atom_channel)

        self.bias_proj = nn.Linear(atom_channel, self.num_head)
        self.dropout = nn.Dropout(model_config.dropout_rate)

        self.out_proj = nn.Linear(atom_channel, atom_channel)
        self.out_dropout = nn.Dropout(model_config.dropout_rate)

    def get_attention_update(self, query_acts, key_acts, key_mask, attention_bias):
        """tbd"""
        B, N1, D = paddle.shape(query_acts)
        _, N2, _ = paddle.shape(key_acts)
        H, d = self.num_head, self.head_dim

        q = self.q_proj(query_acts).reshape([B, N1, H, d]).transpose([0, 2, 1, 3]) # (B, H, N1, d)
        q *= (1 / d ** 0.5)
        k = self.k_proj(key_acts).reshape([B, N2, H, d]).transpose([0, 2, 1, 3])  # (B, H, N2, d)
        v = self.v_proj(key_acts).reshape([B, N2, H, d]).transpose([0, 2, 1, 3])  # (B, H, N2, d)

        attn_bias = self.bias_proj(attention_bias).transpose([0, 3, 1, 2])  # (B, H, N1, N2)

        attn_weights = paddle.matmul(q, k, transpose_y=True)    # (B, H, N1, N2)
        attn_weights += (1 - key_mask).unsqueeze([1, 2]) * (-1e6)   # (B, N2) -> (B, 1, 1, N2)
        attn_weights += attn_bias
        attn_probs = paddle.nn.functional.softmax(attn_weights) # (B, H, N1, N2)
        attn_probs = self.dropout(attn_probs)

        output = paddle.matmul(attn_probs, v) # (B, H, N1, d)
        output = output.transpose([0, 2, 1, 3]).reshape([B, N1, D])
        output = self.out_proj(output)
        return output
  
    def forward(self, query_acts, key_acts, key_mask, attention_bias):
        """
        query_acts: (B, N1, D)
        key_acts: (B, N2, D)
        key_mask: (B, N2)
        attention_bias: (B, N1, N2, D)
        return:
            output: (B, N1, D)
        """
        query_acts = self.q_ln(query_acts)
        key_acts = self.k_ln(key_acts)
        output = self.get_attention_update(query_acts, key_acts, key_mask, attention_bias)
        output = self.out_dropout(output)
        return output


class E3Attention(nn.Layer):
    """Compute self-attention over columns of a 2D input."""
    def __init__(self, model_config, global_config):
        super().__init__()
        self.model_config = model_config

        atom_channel = global_config.atom_channel


        self.num_head = model_config.num_head
        self.head_dim = atom_channel // self.num_head

        self.q_ln = nn.LayerNorm(atom_channel)
        self.k_ln = nn.LayerNorm(atom_channel)

        self.q_proj = nn.Linear(atom_channel, atom_channel)
        self.k1_proj = nn.Linear(atom_channel, atom_channel)
        self.k2_proj = nn.Linear(atom_channel, atom_channel)

        self.bias1_proj = nn.Linear(atom_channel, self.num_head)
        self.bias2_proj = nn.Linear(atom_channel, self.num_head)
        self.dropout = nn.Dropout(model_config.dropout_rate)

    def get_attention_update(self, query_acts, key_acts, value_pos, key_mask, attention_bias):
        """tbd"""
        B, N1, D = paddle.shape(query_acts)
        _, N2, _ = paddle.shape(key_acts)
        H, d = self.num_head, self.head_dim

        q = self.q_proj(query_acts).reshape([B, N1, H, d]).transpose([0, 2, 1, 3]) # (B, H, N1, d)
        q *= (1 / d ** 0.5)
        k1 = self.k1_proj(key_acts).reshape([B, N2, H, d]).transpose([0, 2, 1, 3])  # (B, H, N2, d)
        k2 = self.k2_proj(key_acts).reshape([B, N2, H, d]).transpose([0, 2, 1, 3])  # (B, H, N2, d)

        attn_bias1 = self.bias1_proj(attention_bias).transpose([0, 3, 1, 2])  # (B, H, N1, N2)
        attn_bias2 = self.bias2_proj(attention_bias).transpose([0, 3, 1, 2])  # (B, H, N1, N2)

        ## get attention prob
        attn_weights1 = paddle.matmul(q, k1, transpose_y=True)    # (B, H, N1, N2)
        attn_weights1 += (1 - key_mask).unsqueeze([1, 2]) * (-1e6)   # (B, N2) -> (B, 1, 1, N2)
        attn_weights1 += attn_bias1
        attn_probs = paddle.nn.functional.softmax(attn_weights1) # (B, H, N1, N2)

        attn_weights2 = paddle.matmul(q, k2, transpose_y=True)    # (B, H, N1, N2)
        attn_weights2 += attn_bias2
        attn_weights2 *= key_mask.unsqueeze([1, 2])
        attn_probs *= attn_weights2

        ## update value
        attn_probs = self.dropout(attn_probs).unsqueeze([-2])   # (B, H, N1, 1, N2)
        value_pos = paddle.tile(value_pos.unsqueeze([1]), [1, H, 1, 1, 1])  # (B, H, N1, N2, 3)
        output_pos = paddle.matmul(attn_probs, value_pos) # (B, H, N1, 1, 3)
        output_pos = output_pos.squeeze([-2]).mean([1])   # (B, N1, 3)
        return output_pos
  
    def forward(self, query_acts, key_acts, value_pos, key_mask, attention_bias):
        """
        query_acts: (B, N1, D)
        key_acts: (B, N2, D)
        key_mask: (B, N2)
        value_pos: (B, N1, N2, 3)
        attention_bias: (B, N1, N2, D)
        return:
            output_pos: (B, N1, 3), relative pos
        """
        query_acts = self.q_ln(query_acts)
        key_acts = self.k_ln(key_acts)
        output_pos = self.get_attention_update(
                query_acts, key_acts, value_pos, key_mask, attention_bias)
        return output_pos


class FeedForwardNetwork(nn.Layer):
    """
    FFN for the transformer
    """
    def __init__(self, model_config, input_channel):
        super(FeedForwardNetwork, self).__init__()
        hidden_channel = input_channel * model_config.hidden_factor

        self.ln = nn.LayerNorm(input_channel)
        self.fc1 = nn.Linear(input_channel, hidden_channel)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(model_config.dropout_rate)
        self.fc2 = nn.Linear(hidden_channel, input_channel)
        self.out_dropout = nn.Dropout(model_config.dropout_rate)

    def forward(self, x):
        """
        tbd
        """
        x = self.ln(x)
        x = self.fc2(self.dropout(self.act(self.fc1(x))))
        x = self.out_dropout(x)
        return x


class HelixDockBlock(nn.Layer):
    """
    HelixDockBlock
    """
    def __init__(self, model_config, global_config, tdim, out_ch):
        super().__init__()
        self.model_config = model_config

        atom_channel = global_config.atom_channel
        # edge_channel = global_config.edge_channel
        
        ### cov graph
        self.ligand_cov_pre = nn.Sequential(nn.LayerNorm(atom_channel), nn.Swish())
        self.ligand_cov_gnn = LiteGEMConv(model_config.ligand_cov_gnn, with_efeat=False)
        self.ligand_cov_dropout = nn.Dropout(model_config.dropout_rate)
        self.protein_cov_pre = nn.Sequential(nn.LayerNorm(atom_channel), nn.Swish())
        self.protein_cov_gnn = LiteGEMConv(model_config.protein_cov_gnn, with_efeat=False)
        self.protein_cov_dropout = nn.Dropout(model_config.dropout_rate)

        ### dist rbf
        self.ligand_dist_rbf = RBFEmbedding(['ligand_dist'], atom_channel, model_config.rbf_params)
        self.protein_dist_rbf = RBFEmbedding(['protein_dist'], atom_channel, model_config.rbf_params)
        # added
        self.center_dist_rbf = RBFEmbedding(['ligand_dist'], atom_channel, model_config.rbf_params)

        ### non-cov attention
        self.ligand_attn = CrossAttention(model_config.ligand_attn, global_config)
        self.ligand_ffn = FeedForwardNetwork(model_config.ligand_ffn, atom_channel)

        self.protein_attn = CrossAttention(model_config.protein_attn, global_config)
        self.protein_ffn = FeedForwardNetwork(model_config.protein_ffn, atom_channel)

        self.protein_complex_attention = model_config.get('protein_complex_attention', True) 
        ### e3 attention
        if 'e3_attn' in model_config:
            self.e3_attn = E3Attention(self.model_config.e3_attn, global_config)
            # added
            self.e3_attn_center = E3Attention(self.model_config.e3_attn, global_config)


        if tdim is not None:
            self.temb_proj = nn.Sequential(
                Swish(),
                nn.Linear(tdim, out_ch),
            )

    def _get_flat_unflat_func(self, x):
        """x: (B, N, D)"""
        B, N, D = x.shape
        flat_func = lambda a: a.reshape([B * N, D])
        unflat_func = lambda a: a.reshape([B, N, D])
        return flat_func, unflat_func

    def forward(self, 
            batch, 
            ligand_atom_acts, ligand_bond_acts, ligand_cur_pos,
            protein_atom_acts, protein_bond_acts,
            temb=None, center_cur_pos=None):
        """
        ligand_atom_acts: (B, N, D)
        ligand_bond_acts: (B_E, D)
        protein_atom_acts: (B, N', D)
        protein_bond_acts: (B_E', D)
        """
        ligand_cov_graph = batch['ligand_cov_graph']
        ligand_atom_mask = batch['ligand_atom_mask']
        protein_cov_graph = batch['protein_cov_graph']
        protein_atom_mask = batch['protein_atom_mask']
        if temb is not None:
            protein_atom_pos = batch['protein_atom_pos_zero']
        else:
            protein_atom_pos = batch['protein_atom_pos']
        all_lp_pos = paddle.concat([ligand_cur_pos, protein_atom_pos], 1)    # (B, N + N', 3)

        if temb is not None:
            ligand_atom_acts += self.temb_proj(temb)[:, None, :]

        ### cov graph
        flat_func, unflat_func = self._get_flat_unflat_func(ligand_atom_acts)
        ligand_atom_acts += unflat_func(self.ligand_cov_dropout(
                self.ligand_cov_gnn(
                    ligand_cov_graph, 
                    self.ligand_cov_pre(flat_func(ligand_atom_acts)), 
                    ligand_bond_acts)[0]))

        flat_func, unflat_func = self._get_flat_unflat_func(protein_atom_acts)
        protein_atom_acts += unflat_func(self.protein_cov_dropout(
                self.protein_cov_gnn(
                    protein_cov_graph, 
                    self.protein_cov_pre(flat_func(protein_atom_acts)), 
                    protein_bond_acts)[0]))
        
        ### dist
        ligand_to_lp_dist = distance_matrix(ligand_cur_pos, all_lp_pos)         # (B, N, N + N')
        ligand_to_lp_dist_acts = self.ligand_dist_rbf({'ligand_dist': ligand_to_lp_dist})

        ### non-cov attention
        key_acts = paddle.concat([ligand_atom_acts, protein_atom_acts], 1)  # (B, N+N', D)
        key_mask = paddle.concat([ligand_atom_mask, protein_atom_mask], 1)  # (B, N+N')
        ligand_atom_acts += self.ligand_attn(
                ligand_atom_acts, key_acts, key_mask, ligand_to_lp_dist_acts)
        ligand_atom_acts += self.ligand_ffn(ligand_atom_acts)

        if self.protein_complex_attention:
            protein_to_lp_dist = distance_matrix(protein_atom_pos, all_lp_pos)         # (B, N', N + N')
            protein_to_lp_dist_acts = self.protein_dist_rbf({'protein_dist': protein_to_lp_dist})
            protein_atom_acts += self.protein_attn(
                protein_atom_acts, key_acts, key_mask, protein_to_lp_dist_acts)
        else:
            protein_to_l_dist = distance_matrix(protein_atom_pos, ligand_cur_pos)   # (B, N', N)
            protein_to_l_dist_acts = self.protein_dist_rbf({'protein_dist': protein_to_l_dist})
            protein_atom_acts += self.protein_attn(
                    protein_atom_acts, ligand_atom_acts, ligand_atom_mask, protein_to_l_dist_acts)
        protein_atom_acts += self.protein_ffn(protein_atom_acts)

        ### e3 attention
        if 'e3_attn' in self.model_config:
            key_acts = paddle.concat([ligand_atom_acts, protein_atom_acts], 1)  # (B, N+N', D)
            key_mask = paddle.concat([ligand_atom_mask, protein_atom_mask], 1)  # (B, N+N')
            value_pos = relative_position(ligand_cur_pos, all_lp_pos)   # (B, N, N+N', 3)
            delta_ligand_atom_pos = self.e3_attn(
                    ligand_atom_acts, key_acts, value_pos, key_mask, ligand_to_lp_dist_acts)
            ## global
            if center_cur_pos is not None:
                mask = batch['ligand_atom_mask'].sum(axis=-1).unsqueeze(-1)
                ligand_atom_acts_center = paddle.sum(ligand_atom_acts, axis=1) / mask  #(B, D)
                # center_atom_acts = ligand_atom_acts_center.unsqueeze(1)
                # added4
                center_atom_acts = ligand_atom_acts_center.unsqueeze(1).tile([1, 3, 1])
                ### dist
                center_to_lp_dist = distance_matrix(center_cur_pos, all_lp_pos)         # (B, 3, N+N')
                center_to_lp_dist_acts = self.center_dist_rbf({'ligand_dist': center_to_lp_dist})

                value_center_pos = relative_position(center_cur_pos, all_lp_pos)   # (B, 3, N+N', 3)
                delta_center_pos = self.e3_attn_center(
                    center_atom_acts, key_acts, value_center_pos, key_mask, center_to_lp_dist_acts)
                return ligand_atom_acts, protein_atom_acts, delta_ligand_atom_pos, delta_center_pos
            return ligand_atom_acts, protein_atom_acts, delta_ligand_atom_pos
        
        return ligand_atom_acts, protein_atom_acts


class HelixDockIteraction(nn.Layer):
    """
    HelixDock

    Args:
        model_config(dict): a dict of model configurations.
    """
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config

        atom_channel = self.model_config.atom_channel
        # edge_channel = self.model_config.edge_channel

        self.embedding_layer = EmbeddingLayer(model_config.embedding_layer, model_config)

        self.ligand_atom_ln = nn.LayerNorm(atom_channel)
        self.ligand_atom_drop = nn.Dropout(p=self.model_config.init_dropout_rate)
        self.ligand_edge_drop = nn.Dropout(p=self.model_config.init_dropout_rate)
        self.protein_atom_ln = nn.LayerNorm(atom_channel)
        self.protein_atom_drop = nn.Dropout(p=self.model_config.init_dropout_rate)
        self.protein_edge_drop = nn.Dropout(p=self.model_config.init_dropout_rate)

        self.prev_ligand_atom_ln = nn.LayerNorm(atom_channel)
        self.prev_protein_atom_ln = nn.LayerNorm(atom_channel)

        ## diffusion
        if self.model_config.init_ligand_atom_pos_name == "Diffusion":
            T = self.model_config.T
            ch = self.model_config.ch
            tdim = ch * 4
            self.time_embedding = TimeEmbedding(T, ch, tdim)
        else:
            tdim = None
            ch = None

        self.helixdock_blocks = nn.LayerList()
        for _ in range(self.model_config.helixdock_block_num):
            block = HelixDockBlock(self.model_config.helixdock_block, self.model_config, tdim, ch)
            self.helixdock_blocks.append(block)

        self.helixdock_e3_blocks = nn.LayerList()
        for _ in range(self.model_config.helixdock_e3_block_num):
            block = HelixDockBlock(self.model_config.helixdock_e3_block, self.model_config, tdim, ch)
            self.helixdock_e3_blocks.append(block)

        self.helix_seq_blocks = nn.LayerList()
        for _ in range(self.model_config.helixdock_seq_block_num):
            block = HelixDockBlock(self.model_config.helixdock_seq_block, self.model_config, tdim, ch)
            self.helix_seq_blocks.append(block)

    def to_bytes(self, target_str):
        return sum(list(target_str.encode('ascii'))) % 2
    
    def calculate_unit_dict(self, center_cur_pos):
        """
        tbd
        """
        # center_cur_pos (B, 3, 3)
        center_relative_pos = center_cur_pos - center_cur_pos[:, 0:1, :]
        x1 = center_relative_pos[:, 1:2, :]     # x_axis: (B, 1, 3)
        x2 = center_relative_pos[:, 2:, :]
        z_axis = paddle.cross(x1, x2, axis=2)
        x_norm = paddle.norm(x1, axis=2).unsqueeze(2)
        dx = paddle.divide(x1, x_norm)
        z_norm = paddle.norm(z_axis, axis=2).unsqueeze(2)
        dz = paddle.divide(z_axis, z_norm)
        dy = paddle.cross(dz, dx, axis=2)
        unit_axis = paddle.concat([dx, dy, dz], axis=1)
        return unit_axis

    def update_ligand_relative_pos(self, delta_ligand_atom_pos, unit_axis):
        """
        tbd
        """
        tmp_list = unit_axis.tolist()
        inv_list = []
        for i in tmp_list:
            inv_list.append(np.linalg.pinv(i))
        unit_axis_inv = paddle.to_tensor(inv_list)
        ans1 = paddle.matmul(delta_ligand_atom_pos, unit_axis_inv)
        ans2 = paddle.matmul(delta_ligand_atom_pos, paddle.inverse(unit_axis))
        return paddle.matmul(delta_ligand_atom_pos, unit_axis_inv)

    def forward(self, batch, prev):
        """
        Build the network.
        """
        embed_repr = self.embedding_layer(batch)
        ligand_atom_acts = embed_repr['ligand_atom_acts']
        ligand_bond_acts = embed_repr['ligand_bond_acts']
        protein_atom_acts = embed_repr['protein_atom_acts']
        protein_bond_acts = embed_repr['protein_bond_acts']

        if 'prev_ligand_atom_acts' in prev:
            ligand_atom_acts += self.prev_ligand_atom_ln(prev['prev_ligand_atom_acts'])
            protein_atom_acts += self.prev_protein_atom_ln(prev['prev_protein_atom_acts'])

        ligand_atom_acts = self.ligand_atom_drop(self.ligand_atom_ln(ligand_atom_acts))
        ligand_bond_acts = self.ligand_edge_drop(ligand_bond_acts)
        protein_atom_acts = self.protein_atom_drop(self.protein_atom_ln(protein_atom_acts))
        protein_bond_acts = self.protein_edge_drop(protein_bond_acts)
        if 'prev_ligand_atom_pos' in prev:
            batch['prev_ligand_atom_pos'] = prev['prev_ligand_atom_pos']
        else:
            batch['prev_ligand_atom_pos'] = None
        atom_pos_name = self.model_config.init_ligand_atom_pos_name
        temb = None
        self.use_diffusion = False
        if atom_pos_name == 'zero':
            ligand_cur_pos = batch['ligand_zero_atom_pos']
        elif atom_pos_name == 'Diffusion':
            ligand_cur_pos = batch['ligand_cur_pos']
            time_step = batch['time_step']
            temb = self.time_embedding(time_step)
            self.use_diffusion = True
            self.mean_type = self.model_config.mean_type
        else:
            raise ValueError(atom_pos_name)
        # added
        center_cur_pos = None
        ligand_relative_pos = None
        ligand_pred_pos_list = []
        # added2
        eps_theta_list = []
        for block_i, block in enumerate(self.helixdock_blocks):
            ligand_atom_acts, protein_atom_acts = recompute_wrapper(block, 
                    batch, 
                    ligand_atom_acts, ligand_bond_acts, ligand_cur_pos,
                    protein_atom_acts, protein_bond_acts, temb,
                    is_recompute=self.training)
        for block_i, block in enumerate(self.helixdock_e3_blocks):
            if self.use_diffusion and self.mean_type == 'epsilon':
                eps_theta = 0
                ligand_atom_acts, protein_atom_acts, delta_ligand_atom_pos = recompute_wrapper(block, 
                        batch, 
                        ligand_atom_acts, ligand_bond_acts, ligand_cur_pos,
                        protein_atom_acts, protein_bond_acts,
                        temb, center_cur_pos,
                        is_recompute=self.training)
                ligand_cur_pos += delta_ligand_atom_pos
                eps_theta_list.append(ligand_cur_pos)
                ligand_cur_pos = ligand_cur_pos.detach()
            ## Original
            else:
                ligand_atom_acts, protein_atom_acts, delta_ligand_atom_pos = recompute_wrapper(block, 
                        batch, 
                        ligand_atom_acts, ligand_bond_acts, ligand_cur_pos,
                        protein_atom_acts, protein_bond_acts,
                        temb, center_cur_pos,
                        is_recompute=self.training)
                ligand_cur_pos += delta_ligand_atom_pos
                ligand_pred_pos_list.append(ligand_cur_pos)
                # IMPORTANT: stop_gradient to make the update of ligand_atom_pos stable
                ligand_cur_pos = ligand_cur_pos.detach()
        seq_ligand_atom_acts_list = []
        seq_protein_atom_acts_list = []
        for block_i, block in enumerate(self.helix_seq_blocks):
            if block_i == 0:
                seq_ligand_atom_acts = embed_repr['ligand_atom_acts']
                seq_ligand_bond_acts = embed_repr['ligand_bond_acts']
                seq_protein_atom_acts = embed_repr['protein_atom_acts']
                seq_protein_bond_acts = embed_repr['protein_bond_acts']
            seq_ligand_atom_acts, seq_protein_atom_acts = recompute_wrapper(block, 
                    batch, 
                    seq_ligand_atom_acts, seq_ligand_bond_acts, ligand_cur_pos,
                    seq_protein_atom_acts, seq_protein_bond_acts, temb,
                    is_recompute=self.training)
            seq_ligand_atom_acts_list.append(seq_ligand_atom_acts)
            seq_protein_atom_acts_list.append(seq_protein_atom_acts)
        if len(ligand_pred_pos_list) == 0 :
            ligand_pred_pos_list.append(ligand_cur_pos) 
        results = {
            'ligand_atom_acts': ligand_atom_acts,       # (B, N, D)
            'protein_atom_acts': protein_atom_acts,     # (B, N', D)
            'ligand_pred_pos_list': ligand_pred_pos_list,   # [(B, N, 3)]
            'seq_ligand_atom_acts_list': seq_ligand_atom_acts_list,
            'seq_protein_atom_acts_list': seq_protein_atom_acts_list,
        }
        if len(self.helix_seq_blocks) > 0:
            results['seq_ligand_atom_acts'] = seq_ligand_atom_acts
            results['seq_protein_atom_acts'] = seq_protein_atom_acts
        # added2
        if self.use_diffusion and self.mean_type == 'epsilon':
            results['eps_theta_list'] = eps_theta_list
            results['noise'] = batch['noise']
        return results


class HelixDock(nn.Layer):
    """
    HelixDock

    Args:
        model_config(dict): a dict of model configurations.
    """
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config

        self.helix_dock_iteration = HelixDockIteraction(model_config)
        self.max_recycle_num = model_config.max_recycle_num
        self.global_iter_nums = self._create_global_iter_nums()
        self.train_step_count = 0
        
        logging.info(f'[HelixDock] global_iter_nums: {list(self.global_iter_nums[:8])}')

    def _create_global_iter_nums(self):
        rng = np.random.RandomState(seed=21)
        global_iter_nums = rng.randint(0, self.max_recycle_num + 1, size=[10000000])
        return global_iter_nums

    def _get_prev(self, results):
        return {
            'prev_ligand_atom_acts': results['ligand_atom_acts'].detach(),
            'prev_protein_atom_acts': results['protein_atom_acts'].detach(),
            'prev_ligand_atom_pos': results['ligand_pred_pos_list'][-1].detach(),
        }

    def forward(self, batch):

        if self.training:
            iter_num = self.global_iter_nums[self.train_step_count]
            self.train_step_count += 1
        else:
            iter_num = self.max_recycle_num

        prev = {}
        for iter_i in range(iter_num + 1):
            results = self.helix_dock_iteration(batch, prev)
            prev = self._get_prev(results)
        return results



