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
This is an implementation of CompoundTransformer:
"""
import math
import numpy as np
import paddle
import paddle.nn as nn
from pahelix.networks.compound_encoder import AtomEmbedding, BondEmbedding

class ColumnSelfAttention(nn.Layer):
    """Compute self-attention over columns of a 2D input."""
    def __init__(self,
            embed_dim,
            num_heads,
            dropout_rate=0.0):
        super(ColumnSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.q_e_proj = nn.Linear(embed_dim, embed_dim)
        self.k_e_proj = nn.Linear(embed_dim, embed_dim)
        self.v_e_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def get_attention_probs(self, x, pad_masks, edge_masks):
        """
        x: (B, N, D)
        y: (B, N, N, D)
        pad_masks: (B, N)
        edge_masks: (B, N, N)
        """
        B, N, D = paddle.shape(x)

        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape([B, N, H, d]).transpose([0, 2, 1, 3]) # (B, H, N , d)
        k = self.k_proj(x).reshape([B, N, H, d]).transpose([0, 2, 1, 3])  # (B, H, N, d)

        attn_weights = paddle.matmul(q, k, transpose_y=True)    # (B, H, N, N)
        
        attn_weights += (1 - pad_masks).unsqueeze([1, 2]) * (-1e6)   # (B, N) -> (B, 1, 1, N)
        # add edge mask
        attn_weights += (1 - edge_masks.unsqueeze([1])) * (-1e6) # (B, N, N) -> (B, 1, N, N)

        scaling = 1 / d ** 0.5
        attn_weights *= scaling
        attn_probs = paddle.nn.functional.softmax(attn_weights) # (B, H, N, N)
        return attn_probs

    def get_edge_attention_probs(self, x, y, pad_masks, edge_masks):
        """
        x: (B, N, D)
        y: (B, N, N, D)
        """
        B, N, N, D = paddle.shape(y)

        H, d = self.num_heads, self.head_dim 

        q = self.q_proj(x).reshape([B, N, H, d]).transpose([0, 2, 1, 3]) # (B, H, N , d)
        k_e = self.k_e_proj(y).reshape([B, N, N, H, d]).transpose([0, 3, 1, 2, 4]) #(B, H, N, N, d)

        # q: (B, H, N, d) -> (B, H, N, 1, d)  k_e: (B, H, N, N, d)
        attn_weights = paddle.matmul(q.unsqueeze([3]), k_e, transpose_y=True) #(B, H, N, 1, N)
        attn_weights = attn_weights.reshape([B, H, N, N])

        attn_weights += (1 - pad_masks).unsqueeze([1, 2]) * (-1e6)   # (B, N) -> (B, 1, 1, N)
        # add edge mask
        attn_weights += (1 - edge_masks.unsqueeze([1])) * (-1e6) # (B, N, N) -> (B, 1, N, N)

        scaling = 1 / d ** 0.5
        attn_weights *= scaling
        attn_probs = paddle.nn.functional.softmax(attn_weights) # (B, H, N, N)
        return attn_probs

    def get_attention_update(self, x, y, attn_probs, attn_probs_e):
        """
        x: (B, N, D)
        y: (B, N, N, D)
        attn_probs: (B, H, N, N)
        atten_probs_e : (B, H, N, N)
        """
        B, N, D = paddle.shape(x)
        H, d = self.num_heads, self.head_dim

        v = self.v_proj(x).reshape([B, N, H, d]).transpose([0, 2, 1, 3])  # (B, H, N, d)
        v_e  = self.v_e_proj(y).reshape([B, N, N, H, d]).transpose([0, 3, 1, 2, 4]) # (B, H, N, N, d)

        output = paddle.matmul(attn_probs, v)  # (B, H, N, d)
        output_e = paddle.matmul(attn_probs_e.reshape([B, H, N, 1, N]), v_e) # (B, H, N, 1, d)
        output_e = output_e.reshape([B, H, N, d])
        output = output.transpose([0, 2, 1, 3]).reshape([B, N, D])
        output_e = output_e.transpose([0, 2, 1, 3]).reshape([B, N, D])
        output = output + output_e
        output = self.out_proj(output)
        return output
    
    def forward(self, x, y, pad_masks, edge_masks):
        """
        x: (B, N, D)
        pad_masks: (B, R, C), 0 for padding, 1 for real values

        return:
            output: (B, N, D)
        """
        attn_probs = self.get_attention_probs(x, pad_masks, edge_masks)
        attn_probs = self.dropout(attn_probs)
        attn_probs_e = self.get_edge_attention_probs(x, y, pad_masks, edge_masks)
        attn_probs_e = self.dropout(attn_probs_e)
        output = self.get_attention_update(x, y, attn_probs, attn_probs_e)
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


class EdgeTransformerLayer(nn.Layer):
    """
    Column version of the Axial Transformer
    """
    def __init__(self,
            embed_dim=512,
            ffn_embed_dim=2048,
            n_heads=8,
            dropout_rate=0.1,
            act_dropout_rate=0.1):
        super(EdgeTransformerLayer, self).__init__()


        self.column_layer_norm = nn.LayerNorm(embed_dim)
        self.column_self_attention = ColumnSelfAttention(
                embed_dim, n_heads, dropout_rate=dropout_rate)
        self.column_drop = nn.Dropout(dropout_rate)

        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(
                embed_dim, ffn_embed_dim, dropout_rate=act_dropout_rate)
        self.ffn_drop = nn.Dropout(dropout_rate)

    def forward(self, x, y, pad_masks, edge_masks, return_row_attn=False):
        """
        x: (B, N, D)
        y: (B, N, N, D)
        pad_masks: (B, N)
        edge_masks: (B, N, N)

        return:
            ffn_out: (B, R, C, D)
            ffn_out: (B, N, D)
            //row_attn: (B, H, C, C)
        """
        column_out = self.column_layer_norm(x)
        # column_out_y = self.column_layer_norm(y)

        column_out = self.column_self_attention(column_out, y, pad_masks, edge_masks)
        column_out = self.column_drop(column_out) + x

        ffn_out = self.ffn_layer_norm(column_out)
        ffn_out = self.ffn(ffn_out)
        ffn_out = self.ffn_drop(ffn_out) + column_out

        return ffn_out
  

class CompoundTransformer(nn.Layer):
    """
    Implementation of the compount transformer
    """
    def __init__(self, model_config):
        super(CompoundTransformer, self).__init__()

        self.atom_names = model_config['atom_names']
        self.bond_names = model_config['bond_names']

        self.embed_dim = model_config.get('embed_dim', 512)
        self.init_atom_embedding = AtomEmbedding(self.atom_names, self.embed_dim)
        self.init_bond_embedding = BondEmbedding(self.bond_names, self.embed_dim)
        self.ffn_embed_dim = model_config.get('ffn_embed_dim', 2048)
        self.n_layers = model_config.get('n_layers', 8)
        self.n_heads = model_config.get('n_heads', 8)
        self.dropout_rate = model_config.get('dropout_rate', 0.1)
        self.act_dropout_rate = model_config.get('act_dropout_rate', 0.1)


        self.layer_norm_before = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.transformer_blocks = nn.LayerList()
        for i in range(self.n_layers):
            self.transformer_blocks.append(EdgeTransformerLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=self.ffn_embed_dim,
                    n_heads=self.n_heads,
                    dropout_rate=self.dropout_rate,
                    act_dropout_rate=self.act_dropout_rate,
            ))
        self.layer_norm_after = nn.LayerNorm(self.embed_dim)

        self.apply(self.init_weights)

        self.checkpoints = []
        print('[CompoundTransformer] embed_dim:%s' % self.embed_dim)
        print('[CompoundTransformer] dropout_rate:%s' % self.dropout_rate)
        print('[CompoundTransformer] layer_num:%s' % self.n_layers)
        print('[CompoundTransformer] atom_names:%s' % str(self.atom_names))

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
    
    def get_relative_pos(self, max_len):
        q_ids = paddle.arange(0, max_len)
        k_ids = paddle.arange(0, max_len)
        rel_pos_ids = q_ids.unsqueeze(-1) - paddle.tile(k_ids, (max_len, 1))
        rel_pos_ids = paddle.cast(rel_pos_ids, paddle.int32)
        rel_pos_ids = rel_pos_ids.unsqueeze(0)  # (1, C, C)
        return rel_pos_ids

    def output_dim(self):
        return self.embed_dim

    def forward(self, pad_seq, edge_seq, pad_masks, edge_masks, return_compound=True):
        """
        pad_seq: {feat1:(B, N), feat2:(B, N), ...} number of feat is the number of atom names
        edge_seq: {feat1: (B, N, N), feat2:(B, N, N)} number of feat is the number of bond names
        pad_masks: (B, N), 1 for atom, 0 for padding

        return:
            compound_out: (B, N, D)
            prot_out: (B, C, D)
            row_attn: [(B, H, C, C)]
        """
        atom_embedding = self.init_atom_embedding(pad_seq) # (B, N, D) D= embed_dim
        bond_embedding = self.init_bond_embedding(edge_seq)

        x = self.dropout(self.layer_norm_before(atom_embedding))
        # y = self.dropout(self.layer_norm_before(bond_embedding))

        for block in self.transformer_blocks:
            x = block(x, bond_embedding, pad_masks, edge_masks)
            self.checkpoints.append(x.name)
        compound_out = self.layer_norm_after(x)

        if return_compound:
            return compound_out
        