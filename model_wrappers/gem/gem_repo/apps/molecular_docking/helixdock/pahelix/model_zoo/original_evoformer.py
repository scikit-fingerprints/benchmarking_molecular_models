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
This is an implementation of CompoundTransformer, with concat information, virtual node added and multi hop
information added
"""
import math
import numpy as np
import paddle
import paddle.nn as nn
from paddle.distributed.fleet.utils import recompute
from pahelix.networks.compound_encoder import AtomEmbedding, BondEmbedding, AngleEmbedding
from pahelix.networks.basic_block import Residual

def recompute_wrapper(func, *args, is_recompute=True):
    if is_recompute:
        return recompute(func, *args)
    else:
        return func(*args)




class ColumnSelfAttention(nn.Layer):
    """Compute self-attention over columns of a 2D input."""
    def __init__(self,
            embed_dim,
            num_heads,
            dropout_rate=0.0,
            virtual_node=False,
            no_mask=False):
        super(ColumnSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.q_e_proj = nn.Linear(embed_dim, embed_dim)
        self.k_e_proj = nn.Linear(embed_dim, embed_dim)
        self.v_e_proj = nn.Linear(embed_dim, embed_dim)
        self.concat_mlp = nn.Linear(3 * self.head_dim, self.head_dim)
        self.add_mlp = nn.Linear(self.head_dim, self.head_dim)
        self.virtual_node = virtual_node
        self.pair_proj = nn.Linear(embed_dim, num_heads)

        self.no_mask = no_mask
        if self.no_mask:
            self.edge_bias = nn.Linear(embed_dim, self.num_heads)
        if self.virtual_node:
            self.virtual_node_mlp = nn.Linear(self.head_dim, self.head_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def get_node_edge_attention(self, node_acts, pair_acts, pad_masks, edge_masks):
        """
        node_acts: (B, N, D)
        pair_acts: (B, N, N, D)
        pad_masks: (B, N)
        edge_masks: (B, N, N) 
        """
        # pad_masks = pad_masks[0]
        # edge_masks = edge_masks[0]
        B, N, D = paddle.shape(node_acts)

        H, d = self.num_heads, self.head_dim

        q = self.q_proj(node_acts).reshape([B, N, H, d]).transpose([0, 2, 1, 3]) # (B, H, N , d)
        k_n = self.k_proj(node_acts).reshape([B, N, H, d]).transpose([0, 2, 1, 3])  # (B, H, N, d)
        # k_e = self.k_e_proj(pair_acts).reshape([B, N, N, H, d]).transpose([0, 3, 1, 2, 4]) #(B, H, N, N, d)

        q = q.unsqueeze([3]) # (B, H, N, 1 , d)
        k_n = k_n.unsqueeze([2]) # (B, H, 1, N, d)
        # k = k_n + k_e # (B, H, N, N, d)
        k = k_n
        attn_weights = paddle.matmul(q, k, transpose_y=True)    # (B, H, N, 1, N)

        attn_weights = attn_weights.reshape([B, H, N, N])       # (B, H, N, N)
        attn_weights += (1 - pad_masks).unsqueeze([1, 2]) * (-1e6)   # (B, N) -> (B, 1, 1, N)

        pair_bias = self.pair_proj(pair_acts).transpose([0, 3, 1, 2]) # (B, H, N, N)
        if self.no_mask:
            attn_weights = attn_weights + pair_bias
        else:
            attn_weights += (1 - edge_masks.unsqueeze([1])) * (-1e6) # (B, N, N) -> (B, 1, N, N)

        scaling = 1 / d ** 0.5
        attn_weights *= scaling
        attn_probs = paddle.nn.functional.softmax(attn_weights) # (B, H, N, N)
        # attn_probs = attn_probs.reshape([B, H, N, 1, N])
        return attn_probs

    def get_attention_update(self, node_acts, pair_acts, attn_probs):
        """
        node_acts: (B, N, D)
        pair_acts: (B, N, N, D)
        attn_probs: (B, H, N, 1, N)
        """
        B, N, D = paddle.shape(node_acts)
        H, d = self.num_heads, self.head_dim

        v = self.v_proj(node_acts).reshape([B, N, H, d]).transpose([0, 2, 1, 3])  # (B, H, N, d)
        # v_n = v.unsqueeze([3]) #(B, H, N, 1, d)
        # v_r = v.reshape([B, H, 1, N, d])  
        # v_e = self.v_e_proj(pair_acts).reshape([B, N, N, H, d]).transpose([0, 3, 1, 2, 4]) # (B, H, N, N, d)
        # v_final = self.add_mlp(v_n + v_r + v_e) # (B, H, N, N, d)
        v_final = v

        output = paddle.matmul(attn_probs, v_final) # (B, H, N, 1, d)
        # if self.virtual_node:
        #     v_node = paddle.mean(v_final, axis=-2) # (B, H, N, 1 , d)
        #     v_node_repr = self.virtual_node_mlp(v_node)
        #     v_node_repr = v_node_repr.reshape([B, H, N, 1, d])
        #     output = output + v_node_repr
        output = output.reshape([B, H, N, d])
        output = output.transpose([0, 2, 1, 3]).reshape([B, N, D])
        output = self.out_proj(output)
        return output
    
    def forward(self, node_acts, pair_acts, pad_masks, edge_masks):
        """
        node_acts: (B, N, D)
        pair_acts: (B, N, N, D)
        pad_masks: (B, N), 0 for padding, 1 for real values
        edge_masks: (B, N, N) 0 for padding, 1 for real values
        return:
            output: (B, N, D)
        """
        attn_probs = self.get_node_edge_attention(node_acts, pair_acts, pad_masks, edge_masks)
        attn_probs = self.dropout(attn_probs)
        output = self.get_attention_update(node_acts, pair_acts, attn_probs)
        return output




class FeedForwardNetwork(nn.Layer):
    """
    FFN for the transformer
    """
    def __init__(self, embed_dim, ffn_embed_dim, dropout_rate):
        super().__init__()
        
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        tbd
        """
        out = self.fc2(self.dropout(self.act(self.fc1(x))))
        return out

class OuterProductMean(nn.Layer):
    def __init__(self, embed_dim, pair_embed_dim, dropout_rate):
        super(OuterProductMean, self).__init__()
        inner_dim = 32
        self.fc1 = nn.Linear(embed_dim, inner_dim)
        self.fc2 = nn.Linear(embed_dim, inner_dim)
        self.fc_act = nn.Linear(inner_dim * inner_dim, pair_embed_dim)
    
    def forward(self, node_acts, pad_masks):
        """
        node_acts: (B, N, D)
        pad_masks: (B, N)

        return:
            act: (B, C, C, DP)
        """
        # pad_masks = pad_masks[0]
        pad_masks = pad_masks.unsqueeze(-1)    # (B, N , 1)
        left_act = (pad_masks * self.fc1(node_acts)).unsqueeze(1).transpose([0, 2, 3, 1])   # (B, C, DI, R)
        right_act = (pad_masks * self.fc2(node_acts)).unsqueeze(1).transpose([0, 1, 3, 2])  # (B, R, DI, C)
        B, C, DI, R = paddle.shape(left_act)
        left_act = left_act.reshape([B, C * DI, R])
        right_act = right_act.reshape([B, R, DI * C])
        act = paddle.matmul(left_act, right_act).reshape([B, C, DI, DI, C])   # (B, C, DI, DI, C)
        act = act.transpose([0, 1, 4, 2, 3]).reshape([B, C, C, DI * DI])
        epsilon = 1e-3
        act = self.fc_act(act.astype('float32'))
        return act



class TriangleMultiplication(nn.Layer):
    def __init__(self, pair_embed_dim, dropout_rate, outgoing=True):
        super(TriangleMultiplication, self).__init__()
        self.fc_left = nn.Linear(pair_embed_dim, pair_embed_dim)
        self.fc_right = nn.Linear(pair_embed_dim, pair_embed_dim)
        self.fc_left_gate = nn.Linear(pair_embed_dim, pair_embed_dim)
        self.fc_right_gate = nn.Linear(pair_embed_dim, pair_embed_dim)
        self.outgoing = outgoing
        self.layer_norm = nn.LayerNorm(pair_embed_dim)
        self.fc_out = nn.Linear(pair_embed_dim, pair_embed_dim) 
        self.fc_out_gate = nn.Linear(pair_embed_dim, pair_embed_dim)
    
    def forward(self, pair_acts, pad_masks):
        """
        pair_acts: (B, C, C, D2)
        pad_masks: (B, D), 0 for padding, 1 for real values
        """
        # pad_masks = pad_masks[0]
        pad_masks = pad_masks.unsqueeze(1)
        masks_right = paddle.slice(pad_masks, axes=[1], starts=[0], ends=[1])  # (B, 1, C)
        masks_left = masks_right.transpose([0, 2, 1])                              # (B, C, 1)
        pair_masks = paddle.matmul(masks_left, masks_right).unsqueeze(-1)          # (B, C, C, 1)
        left_acts = pair_masks * self.fc_left(pair_acts)
        right_acts = pair_masks * self.fc_right(pair_acts)
        left_gate = nn.functional.sigmoid(self.fc_left_gate(pair_acts))
        right_gate = nn.functional.sigmoid(self.fc_right_gate(pair_acts))
        out_gate = nn.functional.sigmoid(self.fc_out_gate(pair_acts))
        left_acts = left_acts * left_gate                                          # (B, C, C, D2)
        right_acts = right_acts * right_gate                                       # (B, C, C, D2)

        if self.outgoing:
            left_acts = left_acts.transpose([0, 3, 1, 2])                                   # (B, D2, C, C)
            right_acts = right_acts.transpose([0, 3, 2, 1])                                 # (B, D2, C, C)
            pair_acts = paddle.matmul(left_acts, right_acts).transpose([0, 2, 3, 1])
            #
        else:
            left_acts = left_acts.transpose([0, 3, 2, 1])
            right_acts = right_acts.transpose([0, 3, 1, 2])
            pair_acts = paddle.matmul(left_acts, right_acts).transpose([0, 2, 3, 1])
        
        pair_acts = self.fc_out(self.layer_norm(pair_acts.astype('float32')))
        pair_acts = pair_acts * out_gate

        return pair_acts



class PairAttention(nn.Layer):
    def __init__(self, pair_embed_dim, num_heads, dropout_rate):
        super(PairAttention, self).__init__()
        self.pair_embed_dim = pair_embed_dim
        self.num_heads = num_heads
        self.head_dim = pair_embed_dim // num_heads
        self.dropout_rate = dropout_rate
        self.q_proj = nn.Linear(pair_embed_dim, pair_embed_dim)
        self.k_proj = nn.Linear(pair_embed_dim, pair_embed_dim)
        self.v_proj = nn.Linear(pair_embed_dim, pair_embed_dim)
        self.out_proj = nn.Linear(pair_embed_dim, pair_embed_dim)
        self.out_gate = nn.Linear(pair_embed_dim, pair_embed_dim)
    
    def forward(self, q_data, m_data, bias):
        """
        q_data: (B, C, C, D2)
        m_data: (B, C, C, D2)
        bias: (B, C, C)
        """
        q = self.q_proj(q_data).reshape(shape=[0, 0, 0, self.num_heads, self.head_dim])  # (B, C, C, H, HD)
        k = self.k_proj(m_data).reshape(shape=[0, 0, 0, self.num_heads, self.head_dim])
        v = self.v_proj(m_data).reshape(shape=[0, 0, 0, self.num_heads, self.head_dim])
        q = q.transpose(perm=[0, 1, 3, 2, 4]) * (self.head_dim ** (-0.5))   # (B, C, H, C, HD)
        k = k.transpose(perm=[0, 1, 3, 4, 2])   # (B, C, H, HD, C)
        v = v.transpose(perm=[0, 1, 3, 2, 4])   # (B, C, H, C, HD)
        logits = paddle.matmul(q, k) + bias.unsqueeze([1,1,1])       # (B, C, H, C, C)
        weights = nn.functional.softmax(logits)
        out = paddle.matmul(weights, v).transpose([0, 1, 3, 2, 4])
        out = self.out_proj(out.reshape([0, 0, 0, self.pair_embed_dim]))    # (B, C, C, D2)
        gate = nn.functional.sigmoid(self.out_gate(q_data))
        out = out * gate

        return out


class TriangleAttention(nn.Layer):
    def __init__(self, pair_embed_dim, num_heads, dropout_rate, starting=True):
        super(TriangleAttention, self).__init__()
        self.attn_mod = PairAttention(pair_embed_dim, num_heads, dropout_rate)
        self.starting = starting
    
    def forward(self, pair_acts, pad_masks):
        """
        pair_act: (B, C, C, D2)
        pad_masks: (B, R, C), 1 for padding, 0 for real values
        pad_masks: (B, D), 1 for padding, 0 for real values
        return:
            pair_acts: (B, C, C, D2)
        """

        bias =  (1 - pad_masks) * -1e9 # (B, N)
        if self.starting:
            pair_acts = self.attn_mod(pair_acts, pair_acts, bias)
        else:
            pair_acts = pair_acts.transpose([0, 2, 1, 3])
            pair_acts = self.attn_mod(pair_acts, pair_acts, bias)
            pair_acts = pair_acts.transpose([0, 2, 1, 3])

        return pair_acts


class OriginEvoformerLayer(nn.Layer):
    """
    Column version of the Axial Transformer
    """
    def __init__(self,
            embed_dim=512,
            pair_embed_dim=128,
            ffn_embed_dim=2048,
            n_heads=8,
            dropout_rate=0.1,
            act_dropout_rate=0.1,
            virtual_node=False,
            no_mask=False):
        super(OriginEvoformerLayer, self).__init__()

        self.no_mask = no_mask

        self.node_res = Residual(embed_dim,
                            ColumnSelfAttention(
                            embed_dim,
                            n_heads,
                            dropout_rate=dropout_rate,
                            virtual_node=virtual_node,
                            no_mask=self.no_mask),
                            dropout_rate)

        self.ffn_res = Residual(embed_dim,
                            FeedForwardNetwork(
                            embed_dim, 
                            ffn_embed_dim,
                            dropout_rate=act_dropout_rate),
                            dropout_rate)

        self.outer_product_layer_norm = nn.LayerNorm(embed_dim)
        self.outer_product = OuterProductMean(embed_dim, pair_embed_dim, dropout_rate)
        self.outer_product_drop = nn.Dropout(dropout_rate)

        self.multiply_outgoing_res = Residual(
                pair_embed_dim, 
                TriangleMultiplication(
                    pair_embed_dim,
                    dropout_rate,
                    outgoing=True),
                dropout_rate)

        self.multiply_incoming_res = Residual(
                pair_embed_dim,
                TriangleMultiplication(
                    pair_embed_dim,
                    dropout_rate,
                    outgoing=False),
                dropout_rate)

        self.triangle_attn_starting_res = Residual(
                pair_embed_dim,
                TriangleAttention(
                    pair_embed_dim,
                    n_heads,
                    dropout_rate,
                    starting=True),
                dropout_rate)
                                
        self.triangle_attn_ending_res = Residual(
                pair_embed_dim,
                TriangleAttention(
                    pair_embed_dim,
                    n_heads,
                    dropout_rate,
                    starting=False),
                dropout_rate)

        self.pair_ffn_res = Residual(
                pair_embed_dim,
                FeedForwardNetwork(
                    pair_embed_dim,
                    ffn_embed_dim,
                    dropout_rate), 
                dropout_rate)


    def get_tensor_status(self, x):
        """
        get tensor's mean, min, max and std
        """
        return float(x.mean().numpy()), float(x.min().numpy()), float(x.max().numpy()), float(x.std().numpy())

    def forward(self, node_acts, pair_acts, pad_masks, edge_masks, return_row_attn=False):
        """
        node_acts: (B, N, D)
        pair_acts: (B, N, N, D)
        pad_masks: (B, N)
        edge_masks: (B, N, N)

        return:
            ffn_out: (B, N, D)
            pair_acts: (B, N, D)
        """
        pad_masks = pad_masks[0]
        edge_masks = edge_masks[0]

        node_out = self.node_res(node_acts, pair_acts, pad_masks, edge_masks)
        ffn_out = self.ffn_res(node_out)
        
        outer_res = self.outer_product_layer_norm(ffn_out)   # (B, N, D)
        outer_res = self.outer_product(outer_res, pad_masks) # (B, C, C, D2)
        outer_res = self.outer_product_drop(outer_res) + pair_acts #(B, N, N, D) 

        tri_out_res = self.multiply_outgoing_res(outer_res, pad_masks)
        tri_in_res = self.multiply_incoming_res(tri_out_res, pad_masks)
        tri_start_res = self.triangle_attn_starting_res(tri_in_res, pad_masks)
        tri_end_res = self.triangle_attn_ending_res(tri_start_res, pad_masks)

        pair_acts = self.pair_ffn_res(tri_end_res)

        return ffn_out, pair_acts
  

class OriginEvoformer(nn.Layer):
    """
    Implementation of evoformer for molecule prperty prediction
    """
    def __init__(self, model_config):
        super(OriginEvoformer, self).__init__()

        self.atom_names = model_config['atom_names']
        self.bond_names = model_config['bond_names']

        self.embed_dim = model_config.get('embed_dim', 512)
        self.pair_embed_dim = model_config.get('pair_embed_dim', 128)
        self.max_hop = model_config.get('max_hop', 1)
        self.init_atom_embedding = AtomEmbedding(self.atom_names, self.embed_dim)
        self.init_bond_embedding = BondEmbedding(self.bond_names, self.embed_dim)
        self.init_triangle_embedding = AngleEmbedding(self.embed_dim)
        self.ffn_embed_dim = model_config.get('ffn_embed_dim', 2048)
        self.n_layers = model_config.get('n_layers', 8)
        self.n_heads = model_config.get('n_heads', 8)
        self.dropout_rate = model_config.get('dropout_rate', 0.1)
        self.act_dropout_rate = model_config.get('act_dropout_rate', 0.1)
        self.virtual_node = model_config.get('virtual_node', False)
        self.no_mask = model_config.get('no_mask', False)

        self.layer_norm_before = nn.LayerNorm(self.embed_dim)
        self.pair_left_fc = nn.Linear(self.embed_dim, self.pair_embed_dim)
        self.pair_right_fc = nn.Linear(self.embed_dim, self.pair_embed_dim)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.transformer_blocks = nn.LayerList()
        for i in range(self.n_layers):
            self.transformer_blocks.append(OriginEvoformerLayer(
                    embed_dim=self.embed_dim,
                    pair_embed_dim=self.pair_embed_dim,
                    ffn_embed_dim=self.ffn_embed_dim,
                    n_heads=self.n_heads,
                    dropout_rate=self.dropout_rate,
                    act_dropout_rate=self.act_dropout_rate,
                    virtual_node=self.virtual_node,
                    no_mask=self.no_mask,
            ))
        self.layer_norm_after = nn.LayerNorm(self.embed_dim)

        self.apply(self.init_weights)

        self.checkpoints = []
        print('[OriginEvoformer] embed_dim:%s' % self.embed_dim)
        print('[OriginEvoformer] pair_embed_dim:%s' % self.pair_embed_dim)
        print('[OriginEvoformer] dropout_rate:%s' % self.dropout_rate)
        print('[OriginEvoformer] layer_num:%s' % self.n_layers)
        print('[OriginEvoformer] atom_names:%s' % str(self.atom_names))
        print('[OriginEvoformer] bond_names:%s' % str(self.bond_names))
        print('[OriginEvoformer] virtual_node:%s' % str(self.virtual_node))
        print('[OriginEvoformer] hop number:%s' % str(self.max_hop))
        print('[OriginEvoformer] no_mask:%s' % str(self.no_mask))

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=0.02,
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12
    
    def get_pair_activations(self, seq_embed):
        """tbd"""
        left_single = self.pair_left_fc(seq_embed)
        right_single = self.pair_right_fc(seq_embed) 
        pair_activations = left_single.unsqueeze(1) + right_single.unsqueeze(2) 
        return pair_activations 
    
    def output_dim(self):
        """
        tbd
        """
        return self.embed_dim

    @property
    def graph_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    def get_triangle_matrix(self, dist_matrix):
        """
        tbd
        """
        B, n = dist_matrix.shape[0] , dist_matrix.shape[1] 
        d_ij = dist_matrix.reshape([B, n, n, 1])
        d_ik = dist_matrix.reshape([B, n, 1, n])
        d_jk = dist_matrix.reshape([B, 1, n, n])
        ones = paddle.ones((n, n, n))
        d_ij_broadcast = (d_ij * ones).unsqueeze(-1)
        d_ik_broadcast = (d_ik * ones).unsqueeze(-1)
        d_jk_broadcast = (d_jk * ones).unsqueeze(-1)
        triangle_matrix = paddle.concat([d_ij_broadcast, d_ik_broadcast, d_jk_broadcast], axis=-1)
        return triangle_matrix

    def forward(self, node_feats, edge_feats, pad_masks, edge_masks):
        """
        node_feats: {feat1:(B, N), feat2:(B, N), ...} number of feat is the number of atom names
        edge_feats: {feat1: (B, N, N), feat2:(B, N, N)} number of feat is the number of bond names
        pad_masks: (B, N), 1 for real value, 0 for padding
        edge_masks: (B, N, N) 1 for real value, 0 for padding

        return:
            mol_repr: (B, N, D)
        """
        
        
        node_acts = self.init_atom_embedding(node_feats) # (B, N, D) D= embed_dim
        pair_acts = self.init_bond_embedding(edge_feats)
        

        node_acts = self.dropout(self.layer_norm_before(node_acts))
        for block in self.transformer_blocks:
            # node_acts, pair_acts = block(node_acts, pair_acts, pad_masks, edge_masks)
            node_acts, pair_acts = recompute_wrapper(block, node_acts, pair_acts, [pad_masks], [edge_masks], is_recompute=self.training)
            self.checkpoints.append(node_acts.name)
        mol_repr = self.layer_norm_after(node_acts)

        return mol_repr
        