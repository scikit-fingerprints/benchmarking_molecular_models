# !/usr/bin/env python3                                                                                                                                
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
        protein_feats = {name: batch[f'protein_{name}'] for name in protein_feat_names}  # rename "protein_xx" to "xx"
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


class InvariantPointAttention(nn.Layer):
    """Invariant Point attention module.

    The high-level idea is that this attention module works over a set of points
    and associated orientations in 3D space (e.g. protein residues).

    Each residue outputs a set of queries and keys as points in their local
    reference frame.  The attention is then defined as the euclidean distance
    between the queries and keys in the global frame.

    Jumper et al. (2021) Suppl. Alg. 22 "InvariantPointAttention"
    """
    def __init__(self, model_config, global_config,
                 dist_epsilon=1e-8):
        super(InvariantPointAttention, self).__init__()
        atom_channel = global_config.atom_channel
        # self.channel_num = channel_num
        self.config = model_config
        self.global_config = global_config
        self.dist_epsilon = dist_epsilon

        num_head = self.config.num_head
        num_scalar_qk = self.config.num_scalar_qk
        num_point_qk = self.config.num_point_qk
        num_scalar_v = self.config.num_scalar_v
        num_point_v = self.config.num_point_v
        

        assert num_scalar_qk > 0
        assert num_point_qk > 0
        assert num_point_v > 0
        self.q_ln = nn.LayerNorm(atom_channel)
        self.k_ln = nn.LayerNorm(atom_channel)

        self.q_scalar = nn.Linear(
            atom_channel, num_head * num_scalar_qk)
        self.kv_scalar = nn.Linear(
            atom_channel,
            num_head * (num_scalar_v + num_scalar_qk))

        self.q_point_local = nn.Linear(
            atom_channel, num_head * 3 * num_point_qk)
        self.kv_point_local = nn.Linear(
            atom_channel,
            num_head * 3 * (num_point_qk + num_point_v))

        tpw = np.log(np.exp(1.) - 1.)
        self.trainable_point_weights = paddle.create_parameter(
            [num_head], 'float32',
            default_initializer=nn.initializer.Constant(tpw))
        
        self.bias_proj = nn.Linear(atom_channel, num_head)
        zero_init = self.config.get('zero_init', False)
        if zero_init:
            init_w = nn.initializer.Constant(value=0.0)
        else:
            init_w = nn.initializer.XavierUniform()

        c = num_scalar_v + num_point_v * 4
        self.output_projection = nn.Linear(
            num_head * c, atom_channel,
            weight_attr=paddle.ParamAttr(initializer=init_w))

        self.translation_update = nn.Linear(
            atom_channel, 3,
            weight_attr=paddle.ParamAttr(initializer=init_w))
        
        self.final_act_ffn = FeedForwardNetwork(model_config.final_act_ffn, atom_channel)

    def forward(self, query_act, key_act, key_mask, attention_bias, lig_pos, complex_pos):
        # query_act: [B, N, D]
        # key_act : [B, N+N', D]
        # key_mask : (B, N+N')
        # attention bias : [B, N, N', D]
        # mask: [B, N, 1]
        # lig_pos: (B, N, 3)
        # complex_pos: (B, N+N', 3)
        num_atoms = query_act.shape[1]
        num_atoms_complex = key_act.shape[1]
        num_head = self.config.num_head
        num_scalar_qk = self.config.num_scalar_qk
        num_point_qk = self.config.num_point_qk
        num_scalar_v = self.config.num_scalar_v
        num_point_v = self.config.num_point_v

        query_act = self.q_ln(query_act)
        key_act = self.k_ln(key_act)
        # Construct scalar queries of shape:
        q_scalar = self.q_scalar(query_act)
        q_scalar = paddle.reshape(
            q_scalar, [-1, num_atoms, num_head, num_scalar_qk])         #(B, N, H, n_scalar_qk)

        # Construct scalar keys/values of shape:
        # [batch_size, num_target_residues, num_head, num_points]
        kv_scalar = self.kv_scalar(key_act)
        kv_scalar = paddle.reshape(
            kv_scalar,
            [-1, num_atoms_complex, num_head, num_scalar_v + num_scalar_qk]) 
        k_scalar, v_scalar = paddle.split(
            kv_scalar, [num_scalar_qk, -1], axis=-1)        #   (B, N+N', H, n_scalar_v), (B, N+N', H, n_scalar_qk), 

        # Construct query points of shape:
        # [batch_size, num_atoms, num_head, num_point_qk]
        q_point_local = self.q_point_local(query_act) 
        q_point_local = paddle.split(q_point_local, 3, axis=-1) #[ (B, N, H*n_point_qk) ]*3

        # q_point_global = affine.apply_to_point(q_point_local, extra_dims=1)
        q_point_global = [q + lig_pos[:,:, i].unsqueeze(-1) for i,q in enumerate(q_point_local)]
        q_point = [
            paddle.reshape(x, [-1, num_atoms, num_head, num_point_qk])
            for x in q_point_global]        #[ (B, N, H, n_point_qk)]*3

        # Construct key and value points.
        # Key points shape [batch_size, num_atoms_complex, num_head, num_point_qk]
        # Value points shape [batch_size, num_atoms_complex, num_head, num_point_v]
        kv_point_local = self.kv_point_local(key_act) #[ (B, N+N', H*(n_point_qk + n_point_v))*3]
        kv_point_local = paddle.split(kv_point_local, 3, axis=-1)

        # kv_point_global = affine.apply_to_point(kv_point_local, extra_dims=1)
        kv_point_global = [kv + complex_pos[:,:, i].unsqueeze(-1) for i,kv in enumerate(kv_point_local)]
        kv_point_global = [
            paddle.reshape(x, [-1, num_atoms_complex, num_head, num_point_qk + num_point_v])
            for x in kv_point_global]

        k_point, v_point = list(
            zip(*[
                paddle.split(x, [num_point_qk, -1], axis=-1)
                for x in kv_point_global
            ]))  #[ (B, N+N', H, n_point_qk)]*3 [(B, N+N', H, n_point_v)]*3

        # We assume that all queries and keys come iid from N(0, 1) distribution
        # and compute the variances of the attention logits.
        # Each scalar pair (q, k) contributes Var q*k = 1
        scalar_variance = max(num_scalar_qk, 1) * 1.
        # Each point pair (q, k) contributes Var [0.5 ||q||^2 - <q, k>] = 9 / 2
        point_variance = max(num_point_qk, 1) * 9. / 2

        # Allocate equal variance to scalar, point and attention 2d parts so that
        # the sum is 1.

        num_logit_terms = 3
        scalar_weights = np.sqrt(1.0 / (num_logit_terms * scalar_variance))
        point_weights = np.sqrt(1.0 / (num_logit_terms * point_variance))

        trainable_point_weights = nn.functional.softplus(
            self.trainable_point_weights) #(H)
        point_weights *= paddle.unsqueeze(
            trainable_point_weights, axis=1) #(H, 1)

        # [B, N, H, D] => [B, H, N, D], put head dim first
        q_point = [paddle.transpose(x, [0, 2, 1, 3]) for x in q_point]
        # (B, H, N+N', n_point_qk)
        k_point = [paddle.transpose(x, [0, 2, 1, 3]) for x in k_point]
        v_point = [paddle.transpose(x, [0, 2, 1, 3]) for x in v_point]

        dist2 = [
            paddle.square(paddle.unsqueeze(qx, axis=-2) - \
                          paddle.unsqueeze(kx, axis=-3))
            for qx, kx in zip(q_point, k_point)] # [(B, H, N, N+N', n_point_qk)] * 3
        dist2 = sum(dist2) # (B, H, N, N+N', n_point_qk)

        attn_qk_point = -0.5 * paddle.sum(
            paddle.unsqueeze(point_weights, axis=[1, 2]) * dist2, axis=-1) # [H, 1] => [H, 1, 1, 1] * (B, H, N, N+N', n_point_qk)  ==> (B, H, N, N+N')

        q = paddle.transpose(scalar_weights * q_scalar, [0, 2, 1, 3]) #(B, H, N, n_scalar_qk)
        k = paddle.transpose(k_scalar, [0, 2, 1, 3])    #(B, H, N+N', n_scalar_qk)
        v = paddle.transpose(v_scalar, [0, 2, 1, 3])    #(B, H, N+N', n_scalar_v)
        attn_qk_scalar = paddle.matmul(q, paddle.transpose(k, [0, 1, 3, 2]))    #(B, H, N, N+N')
        attn_logits = attn_qk_scalar + attn_qk_point #(B, H, N, N+N')

        attn_bias = self.bias_proj(attention_bias).transpose([0, 3, 1, 2]) #(B, N, N+N', H) -> (B, H, N, N+N')
        attn_logits += attn_bias
        attn_logits += (1 - key_mask).unsqueeze([1, 2]) * (-1e6) #(B, N+N') -> (B, 1, 1, N+N')

        attn = nn.functional.softmax(attn_logits) #(B, H, N, N+N')

        # o_i^h
        # [batch_size, num_query_residues, num_head, num_head * num_scalar_v]
        result_scalar = paddle.matmul(attn, v) #(B, H, N, num_scalar_v )
        result_scalar = paddle.transpose(result_scalar, [0, 2, 1, 3]) #(B, N, H, num_scalar_v )

        # o_i^{hp}
        # [batch_size, num_query_residues, num_head, num_head * num_point_v]
        result_point_global = [
            paddle.sum(paddle.unsqueeze(attn, -1) * paddle.unsqueeze(vx, -3),
                       axis=-2) for vx in v_point] #(B, H, N, N+N' , 1 ) * (B, H, 1, N+N', n_point_v) -> (B, H, N, N+N', n_point_v) => (B, H, N, n_point_v) 
        result_point_global = [
            paddle.transpose(x, [0, 2, 1, 3]) for x in result_point_global] #(B, N, H, n_point_v)

        # Reshape, global-to-local and save
        result_scalar = paddle.reshape(
            result_scalar, [-1, num_atoms, num_head * num_scalar_v]) #(B, N, H*n_scalar_v)
        result_point_global = [
            paddle.reshape(x, [-1, num_atoms, num_head * num_point_v]) # [(B, N, H*n_point_v)]*3
            for x in result_point_global]
        # result_point_local = affine.invert_point(
            # result_point_global, extra_dims=1)
        result_point_local = [rp - lig_pos[:,:, i].unsqueeze(-1) for i, rp in enumerate(result_point_global)]

        result_point_local_norm = paddle.sqrt(
            self.dist_epsilon + paddle.square(result_point_local[0]) + \
            paddle.square(result_point_local[1]) + \
            paddle.square(result_point_local[2])) #(B, N, H*n_point_v)

        output_features = [result_scalar]
        output_features.extend(result_point_local)
        # output_features.extend(
        #     [result_point_local_norm, result_attention_over_2d])
        output_features.extend(
            [result_point_local_norm])

        final_act = paddle.concat(output_features, axis=-1) #((B, N, H*(n_scalar_v + 4 * n_point_v))
        final_act = self.output_projection(final_act)
        final_act += self.final_act_ffn(final_act)
        translation_update = self.translation_update(final_act) #(B, N, 3)

        return  translation_update


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
    def __init__(self, model_config, global_config):
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
        

    def _get_flat_unflat_func(self, x):
        """x: (B, N, D)"""
        B, N, D = x.shape
        flat_func = lambda a: a.reshape([B * N, D])
        unflat_func = lambda a: a.reshape([B, N, D])
        return flat_func, unflat_func

    def forward(self, 
            batch, 
            ligand_atom_acts, ligand_bond_acts, ligand_cur_pos,
            protein_atom_acts, protein_bond_acts):
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
        protein_atom_pos = batch['protein_atom_pos']
        all_lp_pos = paddle.concat([ligand_cur_pos, protein_atom_pos], 1)    # (B, N + N', 3)
        # all_lp_pos = paddle.concat([prev_pos, protein_atom_pos], 1)    # (B, N + N', 3)


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

        self.helixdock_blocks = nn.LayerList()
        for _ in range(self.model_config.helixdock_block_num):
            block = HelixDockBlock(self.model_config.helixdock_block, self.model_config)
            self.helixdock_blocks.append(block)

        self.helixdock_e3_blocks = nn.LayerList()
        for _ in range(self.model_config.helixdock_e3_block_num):
            block = HelixDockBlock(self.model_config.helixdock_e3_block, self.model_config)
            self.helixdock_e3_blocks.append(block)
        
        self.helixdock_ipa_blocks = nn.LayerList()
        for _ in range(self.model_config.helixdock_ipa_block_num):
            block = HelixDockBlock(self.model_config.helixdock_ipa_block, self.model_config)
            self.helixdock_ipa_blocks.append(block)

        self.helix_seq_blocks = nn.LayerList()
        self.gt_ligand_atom_pos = self.model_config.helixdock_seq_block.get('gt_ligand_atom_pos', False)
        self.bond_info = self.model_config.helixdock_seq_block.get('bond_info', False)
        print('[BondInfo] ', self.bond_info)
        self.drop_gnn = self.model_config.helixdock_seq_block.get('drop_gnn', False)
        print('[drop_gnn] ', self.drop_gnn)
        for _ in range(self.model_config.helixdock_seq_block_num):
            block = HelixDockBlock(self.model_config.helixdock_seq_block, self.model_config)
            self.helix_seq_blocks.append(block)

    def to_bytes(self, target_str):
        return sum(list(target_str.encode('ascii'))) % 2

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
        if atom_pos_name == 'zero':
            ligand_cur_pos = batch['ligand_zero_atom_pos']
        elif atom_pos_name == 'ETKDG':
            ligand_cur_pos = batch['ligand_ETKDG_atom_pos']
        else:
            raise ValueError(atom_pos_name)
        ligand_pred_pos_list = []
        for block_i, block in enumerate(self.helixdock_blocks):
            ligand_atom_acts, protein_atom_acts = recompute_wrapper(block, 
                    batch, 
                    ligand_atom_acts, ligand_bond_acts, ligand_cur_pos,
                    protein_atom_acts, protein_bond_acts,
                    is_recompute=self.training)
        for block_i, block in enumerate(self.helixdock_e3_blocks):
            ligand_atom_acts, protein_atom_acts= recompute_wrapper(block, 
                    batch, 
                    ligand_atom_acts, ligand_bond_acts, ligand_cur_pos,
                    protein_atom_acts, protein_bond_acts,
                    is_recompute=self.training)
        seq_ligand_atom_acts_list = []
        seq_protein_atom_acts_list = []
        for block_i, block in enumerate(self.helix_seq_blocks):
            ligand_atom_acts, protein_atom_acts = recompute_wrapper(block, 
                    batch, 
                    ligand_atom_acts, ligand_bond_acts, ligand_cur_pos,
                    protein_atom_acts, protein_bond_acts,
                    is_recompute=self.training)
            seq_ligand_atom_acts_list.append(ligand_atom_acts)
            seq_protein_atom_acts_list.append(protein_atom_acts)
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
            results['seq_ligand_atom_acts'] = ligand_atom_acts 
            results['seq_protein_atom_acts'] = protein_atom_acts 
        return results


class LiteGEMDock(nn.Layer):
    """
    LiteGEMDock

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




