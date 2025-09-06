#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""only include abstractor and llm"""

from typing import Optional, Union, Callable
import os
import math
from model_config import (
    AbstractorConfig,
    NumberDecoderConfig,
    ResidueEmbeddingConfig,
    HelixProtXConfig,
    HelixProtXOutput,
    EncoderOutput,
    NumberDecoderOutput,
    HelixProtXGenerationOutput
)
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
from paddlenlp.transformers.conversion_utils import (
    StateDictNameMapping,
    init_name_mappings,
)
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
from paddlenlp.transformers.model_outputs import BaseModelOutputWithPooling
from paddlenlp.transformers.model_utils import PretrainedModel, unwrap_model, get_parameter_dtype, no_init_weights, \
    ContextManagers
from paddlenlp.transformers.configuration_utils import PretrainedConfig

SAVE_SUBDIR = {
    'language_model': "language_model",
    'residue_embedding': 'residue_embedding',
    'number_decoder': 'number_decoder',
    'structure_abstractor': 'structure_abstractor',
    'sequence_abstractor': 'sequence_abstractor'
}


class StructureEncoder:
    pass


class SequenceEncoder:
    pass


class LayerNormFp32(nn.LayerNorm):
    """Subclass paddle's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def __init__(self, *args, **kwargs):
        super(LayerNormFp32, self).__init__(*args, **kwargs)

    def forward(self, x):
        # PaddlePaddle's LayerNorm already supports automatic mixed precision training,
        # so we don't need to cast to float32 and back manually.
        return super(LayerNormFp32, self).forward(x)


class NumberEmbedding(nn.Layer):
    """
    Learnable number embeddings. Unseen numbers are interpolated linearly.
    Only create embeddings for [-pi, pi) by default
    """

    def __init__(self, num_embeddings=64, embedding_dim=128, num_range=None, bias=False):
        super(NumberEmbedding, self).__init__()
        self.min, self.max = num_range if num_range else (-math.pi, math.pi)
        self.period = self.max - self.min
        assert self.period > 0
        self.num_embeddings = num_embeddings
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        self.numbers = paddle.tensor.linspace(self.min, self.max, num_embeddings + 1)[:-1]  # do not include self.max
        if bias:
            self.bias = nn.Parameter(paddle.zeros([embedding_dim]))
        else:
            self.bias = None

    def forward(self, number):
        """
        Input:
            - number: paddle tensor of arbitrary shape
        Return:
            - tensor of shape `number.shape + [embedding_dim]`
        interpolation:
            [0 ~ num_embeddings - 1] corresponds to (self.min, ..., self.max)
        normalization:
        """
        # support for non-tensor inputs
        if not isinstance(number, paddle.Tensor):
            number = paddle.to_tensor(number, dtype='float64')
        # if out of range, mod to self.num_range first
        canonical_num = (number - self.min) % self.period + self.min
        # calculate corresponding embed id and weights
        float_idx = (canonical_num - self.min) / self.period * self.num_embeddings
        idx_low = paddle.floor(float_idx).astype('int64')
        idx_up = paddle.ceil(float_idx).astype('int64')
        up_weight = float_idx - idx_low
        low_weight = 1 - up_weight
        idx_low = idx_low % self.num_embeddings
        idx_up = idx_up % self.num_embeddings
        # get weighted embedding, calculate in fp64
        embed_low = self.embed(idx_low)
        embed_up = self.embed(idx_up)
        embedding = embed_low * low_weight[..., None] + embed_up * up_weight[..., None]
        # add bias if self.bias exists
        if self.bias is not None:
            embedding = embedding + self.bias
        return embedding


class ResidueEmbeddingAttention(nn.Layer):
    def __init__(self, config):
        super(ResidueEmbeddingAttention, self).__init__()
        if config.embedding_dim % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size ({}) is not a multiple of the number of attention heads ({})".format(
                    config.embedding_dim, config.num_attention_heads
                )
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.embedding_dim // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_proj = nn.Linear(config.embedding_dim, 3 * self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.out_proj = nn.Linear(self.all_head_size, config.embedding_dim)

    def forward(self, hidden_states):
        """
        hidden_states: [num_residue, num_angles, embedding_dim]
        """
        num_residue, num_angles, embedding_dim = hidden_states.shape
        # get q, k, v
        qkv = self.qkv_proj(hidden_states)  # [num_residue, num_angles, 3 * num_head * head_size]
        qkv = paddle.reshape(qkv, [num_residue, num_angles, 3 * self.num_attention_heads,
                                   self.attention_head_size])  # [num_residue, num_angles, 3 * num_head, head_size]
        # qkv = qkv.reshape(num_residue, num_angles, 3 * self.num_attention_heads, self.attention_head_size)  # [num_residue, num_angles, 3 * num_head, head_size]
        q, k, v = qkv.transpose((0, 2, 1, 3)).split(3, axis=1)  # [num_residue, num_head, num_angles, head_size], each

        # attention scores
        attn_scores = paddle.matmul(q, k.transpose((0, 1, 3, 2)))  # [num_residue, num_head, num_angles, num_angles]
        attn_scores = attn_scores / math.sqrt(self.attention_head_size)

        attn_probs = F.softmax(attn_scores, axis=-1)  # [num_residue, num_head, num_angles, num_angles]
        attn_probs = self.dropout(attn_probs)

        hidden_states = paddle.matmul(attn_probs, v)  # [num_residue, num_head, num_angles, head_size]
        # paddle.reshape(qkv, [num_residue, num_angles, 3 * self.num_attention_heads, self.attention_head_size])
        # reshape and return
        hidden_states = hidden_states.transpose((0, 2, 1, 3))
        hidden_states = paddle.reshape(hidden_states, [num_residue, num_angles, -1])
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class ResidueEmbeddingMLP(nn.Layer):
    def __init__(self, config):
        super(ResidueEmbeddingMLP, self).__init__()
        self.act = nn.Silu()  # PaddlePaddle中的SiLU激活函数
        self.reduce = nn.Linear(config.num_angles, 1)
        self.w1 = nn.Linear(config.embedding_dim, config.intermediate_size)
        self.w2 = nn.Linear(config.intermediate_size, config.output_hidden_size)
        self.w3 = nn.Linear(config.embedding_dim, config.intermediate_size)
        self.ffn_ln = nn.LayerNorm(config.intermediate_size, epsilon=config.layer_norm_eps)

    def forward(self, hidden_states):
        # hidden_states: [num_residue, num_angles, angle_hidden_size]
        # [num_residue, num_angles, angle_hidden_size] ->
        # [num_residue, angle_hidden_size, num_angles] ->
        # [num_residue, angle_hidden_size, 1]
        hidden_states = self.reduce(hidden_states.transpose((0, 2, 1)))
        hidden_states = hidden_states.squeeze(-1)  # [num_residue, angle_hidden_size]

        hidden_states = self.act(self.w1(hidden_states)) * self.w3(hidden_states)
        hidden_states = self.ffn_ln(hidden_states)
        hidden_states = self.w2(hidden_states)
        return hidden_states


class ResidueEmbedding(PretrainedModel):
    config_class = ResidueEmbeddingConfig

    def __init__(self, config: ResidueEmbeddingConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.embed_angles = NumberEmbedding(
            num_embeddings=config.num_embeddings,
            embedding_dim=config.embedding_dim,
            num_range=config.num_range,
            bias=config.bias,
        )
        self.self_attn = ResidueEmbeddingAttention(config)
        self.mlp = ResidueEmbeddingMLP(config)

    @classmethod
    def _get_name_mappings(cls, config):
        mappings: list[StateDictNameMapping] = []
        model_mappings = [
        ]
        '''
            embed_angles.embed.weight torch.Size([64, 128])
            self_attn.qkv_proj.weight torch.Size([384, 128])
            self_attn.qkv_proj.bias torch.Size([384])
            self_attn.out_proj.weight torch.Size([128, 128])
            self_attn.out_proj.bias torch.Size([128])
            mlp.reduce.weight torch.Size([1, 6])
            mlp.reduce.bias torch.Size([1])
            mlp.w1.weight torch.Size([512, 128])
            mlp.w1.bias torch.Size([512])
            mlp.w2.weight torch.Size([4096, 512])
            mlp.w2.bias torch.Size([4096])
            mlp.w3.weight torch.Size([512, 128])
            mlp.w3.bias torch.Size([512])
            mlp.ffn_ln.weight torch.Size([512])
            mlp.ffn_ln.bias torch.Size([512])
        '''
        layer_mappings = [
            # ["embed_angles.embed.weight", None, "transpose"],
            ["embed_angles.embed.weight"],
            ["self_attn.qkv_proj.weight", None, "transpose"],
            ["self_attn.qkv_proj.bias"],
            ["self_attn.out_proj.weight", None, "transpose"],
            ["self_attn.out_proj.bias"],
            ["mlp.reduce.weight", None, "transpose"],
            ["mlp.reduce.bias"],
            ["mlp.w1.weight", None, "transpose"],
            ["mlp.w1.bias"],
            ["mlp.w2.weight", None, "transpose"],
            ["mlp.w2.bias"],
            ["mlp.w3.weight", None, "transpose"],
            ["mlp.w3.bias"],
            ["mlp.ffn_ln.weight"],
            ["mlp.ffn_ln.bias"],
        ]
        model_mappings.extend(layer_mappings)

        init_name_mappings(mappings=model_mappings)

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

    def forward(self, angles):
        """
        angles: [num_residue, num_angles]
        """
        angle_embed = self.embed_angles(angles)  # [num_residue, num_angles, angle_hidden_size]
        angle_embed = self.self_attn(angle_embed)  # [num_residue, num_angles, angle_hidden_size]
        angle_embed = self.mlp(angle_embed)  # [num_residue, text_hidden_size]

        return angle_embed


class NumberDecoderMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.output_size = config.output_size
        self._dtype = self._helper.get_default_dtype()
        assert self._dtype in ['float16', 'float32', 'float64', 'bfloat16', 'float'], f"self._dtype: {self._dtype}"
        self.input_offset = self.create_parameter(shape=[self.hidden_size], dtype=self._dtype)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
        self.out_proj = nn.Linear(self.intermediate_size, self.output_size, bias_attr=True)
        self.norm = LayerNormFp32(config.intermediate_size, epsilon=config.layer_norm_eps)
        self.act_fn = nn.Silu()
        self.output_scaling = config.output_scaling
        # self.output_scaling = nn.Parameter(torch.tensor([math.pi, math.pi, 0.6, 0.5, 0.3, 0.3]), requires_grad=True)
        # self.output_offset = nn.Parameter(torch.tensor([0, 1, math.pi, 1.9, 2.05, 2.1]), requires_grad=True)

    def forward(self, hidden_states):
        """
        perform calculations in fp32
        give outputs in fp32 for better precision
        """
        input_dtype = hidden_states.dtype  # not used
        hidden_states = hidden_states + self.input_offset  # to remove potential effect of [NUM] token
        hidden_states = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        hidden_states = self.norm(hidden_states)  # /sqrt(intermediate_size) ?
        # hidden_states = nn.functional.linear(hidden_states.float(), self.out_proj.weight.float(), self.out_proj.bias.float())
        hidden_states = nn.functional.linear(hidden_states, self.out_proj.weight, self.out_proj.bias)

        hidden_states = hidden_states * self.output_scaling  # + self.output_offset.float()

        return hidden_states


class NumberDecoder(PretrainedModel):
    config_class = NumberDecoderConfig

    def __init__(self, config, **kwargs):
        super(NumberDecoder, self).__init__(config, **kwargs)
        self.config = config
        self.loss_scaling = config.loss_scaling
        self.mlp = NumberDecoderMLP(config)
        self.loss_fn = NumberDecoderLoss()

    @classmethod
    def _get_name_mappings(cls, config):
        mappings: list[StateDictNameMapping] = []
        model_mappings = [
        ]
        layer_mappings = [
            ["mlp.input_offset"],
            ["mlp.gate_proj.weight", None, "transpose"],
            ["mlp.up_proj.weight", None, "transpose"],
            ["mlp.out_proj.weight", None, "transpose"],
            ["mlp.out_proj.bias"],
            ["mlp.norm.weight"],
            ["mlp.norm.bias"],
        ]
        model_mappings.extend(layer_mappings)
        init_name_mappings(mappings=model_mappings)
        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

    def forward(self, hidden_states, labels=None):
        # hidden_states: [num_residue, hidden_size]
        # labels: [num_residue, num_angles]

        pred_num = self.mlp(hidden_states)  # should be fp32
        loss = None
        range_loss = None
        rad_loss = None
        rad_loss_by_angle_type = None

        if labels is not None:
            # 假设labels是fp32类型的张量
            loss, range_loss, rad_loss, rad_loss_by_angle_type = self.loss_fn(pred_num, labels)
            loss = loss * self.loss_scaling
            loss.to(dtype=self._dtype)  # 转换为模型的dtype

        return NumberDecoderOutput(
            pred_num=pred_num,
            loss=loss,
            range_loss=range_loss,
            rad_loss=rad_loss,
            rad_loss_by_angle_type=rad_loss_by_angle_type,
        )


class NumberDecoderLoss(nn.Layer):
    def __init__(self, num_range=(-math.pi, math.pi), eps=0.9, range_penalty=1.0, angle_weights=[1, 1, 1, 1, 1, 1]):
        super(NumberDecoderLoss, self).__init__()
        self.min, self.max = num_range
        self.period = self.max - self.min
        self.eps = eps  # 0~9: 0.5
        self.range_penalty = range_penalty
        self.angle_weights = paddle.to_tensor(angle_weights, dtype=paddle.float32)

    def forward(self, pred, label):
        """
        Loss can be divided into 2 parts:
        - "out of range" loss
        - radian loss, considers periodicity
        """
        pred.to(dtype=paddle.float32)
        label.to(dtype=paddle.float32)

        # out-of-range loss
        overflow_idx = pred > (self.max + self.eps)
        underflow_idx = pred < (self.min - self.eps)
        overflow_loss = paddle.zeros([1], dtype=paddle.float32)
        underflow_loss = paddle.zeros([1], dtype=paddle.float32)

        if overflow_idx.any():
            overflow_loss = (pred[overflow_idx] - self.max).mean()
            overflow_loss = overflow_loss * (overflow_idx.sum() / pred.numel())
        if underflow_idx.any():
            underflow_loss = (self.min - pred[underflow_idx]).mean()
            underflow_loss = underflow_loss * (underflow_idx.sum() / pred.numel())
        range_loss = overflow_loss + underflow_loss

        # radian loss
        diff = (pred - label).abs() % self.period
        more_than_half_period_idx = diff > (self.period / 2)
        diff[more_than_half_period_idx] = self.period - diff[more_than_half_period_idx]
        if diff.shape[0] > 1:
            # diff = diff.reshape(-1, diff.shape[-1])
            diff = paddle.reshape(diff, [-1, diff.shape[-1]])
        rad_loss_by_angle_type = paddle.mean(diff, axis=0)
        weight = self.angle_weights
        rad_loss = paddle.mean(rad_loss_by_angle_type * weight)
        total_loss = range_loss * self.range_penalty + rad_loss

        return total_loss, range_loss, rad_loss, rad_loss_by_angle_type


class AbstractorAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        self.num_tokens = config.num_tokens
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """
        reshape to multiple heads and trasnpose
        `x`: [batch_size, seq_len, all_head_size]
        """
        new_x_shape = x.shape[:-1] + [self.num_attention_heads,
                                      self.attention_head_size]  # [batch_size, seq_len, num_attention_heads, attention_head_size]
        x = x.reshape(new_x_shape)
        return x.transpose((0, 2, 1, 3))  # [batch_size, num_attention_heads, seq_len, attention_head_size]

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask, head_mask=None):
        """
        `hidden_states`: [batch_size, num_tokens, hidden_size], as query
        `encoder_hidden_states`: [batch_size, seq_len, encoder_hidden_size], as key and value
        `encoder_attention_mask`: [batch_size, num_attention_heads, num_tokens, seq_len]
        `attention_mask`: [batch_size, num_attention_heads, num_tokens, seq_len]
        """
        query_layer = self.transpose_for_scores(
            self.query(hidden_states))  # [batch_size, num_attention_heads, num_tokens, attention_head_size]
        key_layer = self.transpose_for_scores(
            self.key(encoder_hidden_states))  # [batch_size, num_attention_heads, seq_len, attention_head_size]
        value_layer = self.transpose_for_scores(
            self.value(encoder_hidden_states))  # [batch_size, num_attention_heads, seq_len, attention_head_size]

        # encoder_attention_mask = encoder_attention_mask[:, None, None, :].expand((-1, self.num_attention_heads, self.num_tokens, -1))
        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = paddle.matmul(query_layer, key_layer.transpose(
            (0, 1, 3, 2)))  # [batch_size, num_attention_heads, num_tokens, seq_len]

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if encoder_attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            # attention_scores = attention_scores + attention_mask
            # attention_scores.masked_fill_(encoder_attention_mask==0, float("-inf")) # inplace
            # attention_scores = attention_scores.masked_fill(encoder_attention_mask==0, float("-inf")) # out-of-palce
            attention_scores = paddle.masked_fill(attention_scores, encoder_attention_mask == 0, float("-inf"))

        # Normalize the attention scores to probabilities.
        attention_probs = paddle.nn.functional.softmax(attention_scores, axis=-1, dtype='float32')
        attention_probs.to(dtype=value_layer.dtype)

        # if self.save_attention:
        #     self.save_attention_map(attention_probs)
        #     attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = paddle.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.transpose((0, 2, 1, 3)).contiguous()
        new_context_layer_shape = context_layer.shape[:-2] + [self.all_head_size, ]
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer
        # outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # outputs = outputs + (past_key_value,)
        # return outputs


class AbstractorMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.act = nn.Silu()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_ln = LayerNormFp32(config.intermediate_size, epsilon=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.act(self.w1(hidden_states)) * self.w3(hidden_states)
        hidden_states = self.ffn_ln(hidden_states)
        hidden_states = self.w2(hidden_states)
        return hidden_states


class AbstractorCrossOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = LayerNormFp32(config.hidden_size, epsilon=config.layer_norm_eps)
        self.mlp = AbstractorMLP(config)

    def forward(self, attention_outputs, hidden_states):
        hidden_states = hidden_states + self.linear(attention_outputs)
        hidden_states = hidden_states + self.mlp(self.norm(hidden_states))
        return hidden_states


class AbstractorLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.norm_q = LayerNormFp32(config.hidden_size, epsilon=config.layer_norm_eps)
        self.norm_kv = LayerNormFp32(config.encoder_hidden_size, epsilon=config.layer_norm_eps)
        self.attention = AbstractorAttention(config)
        self.output = AbstractorCrossOutput(config)

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask):
        # normalize inputs
        hidden_states = self.norm_q(hidden_states)
        encoder_hidden_states = self.norm_kv(encoder_hidden_states)
        # cross attention
        attention_outputs = self.attention(hidden_states, encoder_hidden_states, encoder_attention_mask)
        # cross output
        outputs = self.output(attention_outputs, hidden_states)
        return outputs


class AbstractorPooling(nn.Layer):
    def __init__(self, config, text_hidden_size):
        super().__init__()
        self.norm = LayerNormFp32(text_hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, x):
        x = paddle.mean(x, axis=1, keepdim=True)  # reduce number of tokens to 1
        return self.norm(x)


class Abstractor(PretrainedModel):
    """to transform structure embeddings of varied length to fixed length tokens"""
    config_class = AbstractorConfig

    def __init__(self, config: AbstractorConfig, text_hidden_size, **kwargs):
        """
        `text_hidden_size`: dim of text embedding
        """
        super().__init__(config, **kwargs)
        self.config = config
        self._dtype = self._helper.get_default_dtype()
        assert self._dtype in ['float16', 'float32', 'float64', 'bfloat16', 'float'], f"self._dtype: {self._dtype}"
        # self.token_embeds = nn.Parameter(torch.randn(1, config.num_tokens, config.hidden_size))
        self.token_embeds = self.create_parameter(shape=[1, config.num_tokens, config.hidden_size],
                                                  dtype=self._dtype)  # paddle.base.data_feeder.convert_dtype
        self.layers = nn.LayerList([AbstractorLayer(config) for _ in range(
            config.num_hidden_layers)])  # to align structure embed towards fixed length tokens
        self.proj = nn.Linear(config.hidden_size,
                              text_hidden_size)  # project dimension of struc embed to match that of text embed
        # self.struc_eos = nn.Parameter(torch.randn(1, 1, text_hidden_size)) # to separate structure and text embeddings
        assert self._dtype in ['float16', 'float32', 'float64', 'bfloat16', 'float'], f"self._dtype: {self._dtype}"
        self.struc_eos = self.create_parameter(shape=[1, 1, text_hidden_size], dtype=self._dtype)
        self.pool = AbstractorPooling(config, text_hidden_size)

    def expand_mask(self, raw_mask):
        """
        `raw_mask`: [batch_size, seq_len]
        return: [batch_size, num_attention_heads, num_tokens, seq_len]
        """
        return raw_mask[:, None, None, :].expand((-1, self.config.num_attention_heads, self.config.num_tokens, -1))

    def forward(self, encoder_hidden_states, encoder_attention_mask, return_all_hidden_states=False,
                return_pooled_output=False):
        """
        perform cross-attention between structure embed and fixed length embed
            query: structure embed; key, value: fixed length tokens
        `encoder_hidden_states`: features returned by StructureEncoder, [batch_size, seq_len, node_features]
        `encoder_attention_mask`: mask padded positions in encoder_hidden_states, [batch_size, num_attention_heads, num_tokens, seq_len]
        `return_hidden_states`: output list of hidden_states of each layer
        """
        # perform cross attention
        batch_size = encoder_hidden_states.shape[0]
        hidden_states = self.token_embeds.expand((batch_size, -1, -1))
        all_hidden_states = None
        if return_all_hidden_states:
            all_hidden_states = ()
            all_hidden_states += (hidden_states,)
        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states, encoder_attention_mask)
            if return_all_hidden_states:
                all_hidden_states += (hidden_states,)
        # hidden_states: [batch_size, num_tokens, hidden_size]
        # project onto text_hidden_size
        hidden_states = self.proj(hidden_states)
        # TODO: add potential pooling
        if return_pooled_output:
            pooled_output = self.pool(hidden_states)
            pooled_output = paddle.concat([pooled_output, self.struc_eos.expand((batch_size, -1, -1))], axis=1)
        else:
            pooled_output = None
        # add eos token
        hidden_states = paddle.concat([hidden_states, self.struc_eos.expand((batch_size, -1, -1))], axis=1)

        # return
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,  # [batch_size, num_tokens + 1, text_hidden_size]
            pooler_output=pooled_output,  # [batch_size, 1 + 1, text_hidden_siz]
            hidden_states=all_hidden_states,  # [num_layers, batch_size, num_tokens, hidden_size]
        )

    @classmethod
    def _get_name_mappings(cls, config):
        mappings: list[StateDictNameMapping] = []
        model_mappings = []
        layer_mappings = [
            ['token_embeds'],
            ['struc_eos'],
            ['layers.0.norm_q.weight'],
            ['layers.0.norm_q.bias'],
            ['layers.0.norm_kv.weight'],
            ['layers.0.norm_kv.bias'],
            ['layers.0.attention.query.weight', None, 'transpose'],
            ['layers.0.attention.query.bias'],
            ['layers.0.attention.key.weight', None, 'transpose'],
            ['layers.0.attention.key.bias'],
            ['layers.0.attention.value.weight', None, 'transpose'],
            ['layers.0.attention.value.bias'],
            ['layers.0.output.linear.weight', None, 'transpose'],
            ['layers.0.output.linear.bias'],
            ['layers.0.output.norm.weight'],
            ['layers.0.output.norm.bias'],
            ['layers.0.output.mlp.w1.weight', None, 'transpose'],
            ['layers.0.output.mlp.w1.bias'],
            ['layers.0.output.mlp.w2.weight', None, 'transpose'],
            ['layers.0.output.mlp.w2.bias'],
            ['layers.0.output.mlp.w3.weight', None, 'transpose'],
            ['layers.0.output.mlp.w3.bias'],
            ['layers.0.output.mlp.ffn_ln.weight'],
            ['layers.0.output.mlp.ffn_ln.bias'],
            ['proj.weight', None, 'transpose'],
            ['proj.bias'],
            ['pool.norm.weight'],
            ['pool.norm.bias']
        ]
        model_mappings.extend(layer_mappings)
        init_name_mappings(mappings=model_mappings)
        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings


class HelixProtX(PretrainedModel):
    config_class = HelixProtXConfig
    main_input_name = "input_ids"

    def __init__(self, config: HelixProtXConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.structure_abstractor = Abstractor(
            config.structure_abstractor_config,
            config.text_config.hidden_size
        )
        self.sequence_abstractor = Abstractor(
            config.sequence_abstractor_config,
            config.text_config.hidden_size
        )
        self.residue_embedding = ResidueEmbedding(config.residue_embedding_config)
        self.number_decoder = NumberDecoder(config.number_decoder_config)
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.num_struc_token = config.structure_abstractor_config.num_tokens + 1
        self.num_seq_token = config.sequence_abstractor_config.num_tokens + 1
        # TODO: self.tokenizer

    def _get_input_embeddings(
            self,
            input_ids=None,
            attention_mask=None,
            angle_mask=None,
            angle_labels=None,
            structure_encoder_outputs: Optional[EncoderOutput] = None,
            struc_idx=[],
            sequence_encoder_outputs: Optional[EncoderOutput] = None,
            seq_idx=[],
            angle_idx=[],
            input_embeds=None,
            augment_noise_scale=0
    ):
        # build input embeddings
        if input_embeds is None:
            # process encoder features with abstractor
            if structure_encoder_outputs is not None:
                struc_embeds = structure_encoder_outputs.encoder_hidden_states
                struc_attention_mask = structure_encoder_outputs.encoder_attention_mask  # raw mask: [batch_size, seq_len]
                struc_attention_mask = self.structure_abstractor.expand_mask(
                    struc_attention_mask)  # mask for AbstractorAttention.forward(): [batch_size, num_attention_heads, num_tokens, seq_len]
                structure_abstractor_outputs = self.structure_abstractor(struc_embeds, struc_attention_mask)
                structure_query_features = structure_abstractor_outputs.last_hidden_state  # [batch_size, num_tokens + 1, hidden_size]

            if sequence_encoder_outputs is not None:
                seq_embeds = sequence_encoder_outputs.encoder_hidden_states
                seq_attention_mask = sequence_encoder_outputs.encoder_attention_mask
                seq_attention_mask = self.sequence_abstractor.expand_mask(
                    seq_attention_mask)  # mask for AbstractorAttention.forward(): [batch_size, num_attention_heads, num_tokens, seq_len]
                sequence_abstractor_outputs = self.sequence_abstractor(seq_embeds, seq_attention_mask)
                sequence_query_features = sequence_abstractor_outputs.last_hidden_state  # [batch_size, num_tokens + 1, hidden_size]

            # get text embedding
            text_tokens_ = input_ids.clone()
            batch_size = input_ids.shape[0]

            text_tokens_[text_tokens_ < 0] = 1  # Not used
            text_embeds = self.language_model.get_input_embeddings()(text_tokens_)  # Temporally Embedding
            if hasattr(self.language_model, 'transformer') and hasattr(self.language_model.transformer,
                                                                       'word_embeddings_layernorm'):
                text_embeds = self.language_model.transformer.word_embeddings_layernorm(text_embeds)

            text_chunk_embeds = []
            struc_num = 0
            seq_num = 0
            for b in range(batch_size):
                result = []
                if b in struc_idx:
                    result.append(structure_query_features[struc_num])
                    struc_num += 1
                    result.append(text_embeds[b, self.num_struc_token:])
                elif b in seq_idx:
                    result.append(sequence_query_features[seq_num])
                    seq_num += 1
                    result.append(text_embeds[b, self.num_seq_token:])
                else:
                    result.append(text_embeds[b, :])
                text_chunk_embeds.append(paddle.concat(result, axis=0))

            # Actual Input Embeddings
            input_embeds = paddle.stack(text_chunk_embeds, axis=0)
            # Add number embeddings
            if len(angle_idx) > 0:
                input_embeds[angle_mask] = input_embeds[angle_mask] + self.residue_embedding(angle_labels)

        # Add augment noise to input embeds
        if augment_noise_scale > 0:
            noise = paddle.randn_like(input_embeds)
            input_embeds = input_embeds + augment_noise_scale * noise

        return input_embeds

    def get_input_embeddings(
            self,
            input_ids=None,
            attention_mask=None,
            angle_mask=None,
            angle_labels=None,
            structure_encoder_outputs: Optional[EncoderOutput] = None,
            struc_idx=[],
            sequence_encoder_outputs: Optional[EncoderOutput] = None,
            seq_idx=[],
            angle_idx=[],
            input_embeds=None,
            augment_noise_scale=0
    ):
        # build input embeddings
        if input_embeds is None:
            # process encoder features with abstractor
            if structure_encoder_outputs is not None:
                struc_embeds = structure_encoder_outputs.encoder_hidden_states
                struc_attention_mask = structure_encoder_outputs.encoder_attention_mask  # raw mask: [batch_size, seq_len]
                struc_attention_mask = self.structure_abstractor.expand_mask(
                    struc_attention_mask)  # mask for AbstractorAttention.forward(): [batch_size, num_attention_heads, num_tokens, seq_len]
                structure_abstractor_outputs = self.structure_abstractor(struc_embeds, struc_attention_mask)
                structure_query_features = structure_abstractor_outputs.last_hidden_state  # [batch_size, num_tokens + 1, hidden_size]

            if sequence_encoder_outputs is not None:
                seq_embeds = sequence_encoder_outputs.encoder_hidden_states
                seq_attention_mask = sequence_encoder_outputs.encoder_attention_mask
                seq_attention_mask = self.sequence_abstractor.expand_mask(
                    seq_attention_mask)  # mask for AbstractorAttention.forward(): [batch_size, num_attention_heads, num_tokens, seq_len]
                sequence_abstractor_outputs = self.sequence_abstractor(seq_embeds, seq_attention_mask)
                sequence_query_features = sequence_abstractor_outputs.last_hidden_state  # [batch_size, num_tokens + 1, hidden_size]

            # get text embedding
            text_tokens_ = input_ids.clone()
            batch_size = input_ids.shape[0]

            text_tokens_[text_tokens_ < 0] = 1  # Not used
            text_embeds = self.language_model.get_input_embeddings()(text_tokens_)  # Temporally Embedding
            if hasattr(self.language_model, 'transformer') and hasattr(self.language_model.transformer,
                                                                       'word_embeddings_layernorm'):
                text_embeds = self.language_model.transformer.word_embeddings_layernorm(text_embeds)

            text_chunk_embeds = []
            struc_num = 0
            seq_num = 0
            # TODO: add support for left padding
            for b in range(batch_size):
                result = []
                attn_start_idx = attention_mask[b].nonzero()[0]
                if attn_start_idx > 0:
                    result.append(text_embeds[b, :attn_start_idx])
                if b in struc_idx:
                    result.append(structure_query_features[struc_num])
                    struc_num += 1
                    result.append(text_embeds[b, attn_start_idx + self.num_struc_token:])
                elif b in seq_idx:
                    result.append(sequence_query_features[seq_num])
                    seq_num += 1
                    result.append(text_embeds[b, attn_start_idx + self.num_seq_token:])
                else:
                    result.append(text_embeds[b, attn_start_idx:])
                text_chunk_embeds.append(paddle.concat(result, axis=0))

            # Actual Input Embeddings
            input_embeds = paddle.stack(text_chunk_embeds, axis=0)
            # Add number embeddings
            if len(angle_idx) > 0:
                input_embeds[angle_mask] = input_embeds[angle_mask] + self.residue_embedding(angle_labels)

        # Add augment noise to input embeds
        if augment_noise_scale > 0:
            noise = paddle.randn_like(input_embeds)
            input_embeds = input_embeds + augment_noise_scale * noise

        return input_embeds

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            angle_labels=None,
            angle_mask=None,
            label_ids=None,
            structure_encoder_outputs: Optional[EncoderOutput] = None,
            struc_idx=[],
            sequence_encoder_outputs: Optional[EncoderOutput] = None,
            seq_idx=[],
            angle_idx=[],
            input_embeds=None,
            augment_noise_scale=0,
            output_hidden_states: Optional[bool] = None,
            past_key_values=None,
            use_cache=False
    ) -> HelixProtXOutput:
        # build input embeddings
        input_embeds = self.get_input_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            angle_mask=angle_mask,
            angle_labels=angle_labels,
            structure_encoder_outputs=structure_encoder_outputs,
            struc_idx=struc_idx,
            sequence_encoder_outputs=sequence_encoder_outputs,
            seq_idx=seq_idx,
            angle_idx=angle_idx,
            input_embeds=input_embeds,
            augment_noise_scale=0,
        )
        # Forward into base LLM
        if output_hidden_states is None:
            output_hidden_states = (len(angle_idx) > 0)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=label_ids,
            return_dict=True,
            past_key_values=past_key_values,
            output_attentions=self.config.output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache
        )

        number_outputs = None
        if len(angle_idx) > 0:
            if isinstance(angle_mask, tuple):
                angle_batch_idx, angle_pos_idx = angle_mask
            else:
                angle_batch_idx, angle_pos_idx = angle_mask.nonzero(as_tuple=True)
            angle_pos_idx = angle_pos_idx - 1  # offset for next residue prediction
            angle_mask_with_offset = (angle_batch_idx, angle_pos_idx)
            last_hidden_state = outputs.hidden_states[-1]
            # BUG: angle_pos_idx cannot be -1
            # BUG: during kv cache, last_hidden_state has len equal to input_embeds (instead of the entire sequence)
            number_hidden_states = last_hidden_state[angle_batch_idx, angle_pos_idx]
            number_outputs = self.number_decoder(number_hidden_states, angle_labels)

        return HelixProtXOutput(
            language_model_outputs=outputs,
            number_decoder_outputs=number_outputs,
        )

    @paddle.no_grad()
    def _generate(
            self,
            input_ids: Optional[paddle.Tensor] = None,
            attention_mask: Optional[paddle.Tensor] = None,
            structure_encoder_outputs: Optional[EncoderOutput] = None,
            struc_idx=[],
            sequence_encoder_outputs: Optional[EncoderOutput] = None,
            seq_idx=[],
            output_hidden_states=False,
            input_embeds=None,
            **generate_kwargs,
    ):
        # build input embeddings
        input_embeds = self.get_input_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            angle_mask=angle_mask,
            structure_encoder_outputs=structure_encoder_outputs,
            struc_idx=struc_idx,
            sequence_encoder_outputs=sequence_encoder_outputs,
            seq_idx=seq_idx,
            angle_idx=angle_idx,
            input_embeds=input_embeds,
            augment_noise_scale=0,
        )

        generation_config = generate_kwargs.pop('generation_config', None)
        # Forward into GPT

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            generation_config=generation_config,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            **generate_kwargs,
        )
        return outputs

    @paddle.no_grad()
    def generate(
            self,
            input_ids: Optional[paddle.Tensor] = None,
            attention_mask: Optional[paddle.Tensor] = None,
            structure_encoder_outputs: Optional[EncoderOutput] = None,
            struc_idx=[],
            sequence_encoder_outputs: Optional[EncoderOutput] = None,
            seq_idx=[],
            input_embeds=None,
            **generate_kwargs,
    ) -> HelixProtXGenerationOutput:
        """
        generation with kv cache
        performs generation of structure
        assumes batch size 1
        assumes seq_to_func or func_to_seq
        """
        # check padding side
        if attention_mask is None:
            attention_mask = paddle.ones_like(input_ids)
        if (attention_mask[:, 0] == 0).any():
            print("Warning: left padding detected, should use right padding for generation.")
        # build initial input embeddings
        input_embeds = self.get_input_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            structure_encoder_outputs=structure_encoder_outputs,
            struc_idx=struc_idx,
            sequence_encoder_outputs=sequence_encoder_outputs,
            seq_idx=seq_idx,
            input_embeds=input_embeds,
            augment_noise_scale=0,
        )
        batch_size = input_ids.shape[0]
        batch_with_angle = list(range(batch_size))
        pred_angles = []
        generated_tokens = 0
        past_key_values = None
        eos_id = generate_kwargs.pop('eos_token_id', 2)
        num_id = generate_kwargs.pop('num_token_id', 100295)
        max_new_tokens = generate_kwargs.pop('max_new_tokens', 100)
        do_sample = generate_kwargs.pop('do_sample', False)
        top_p = generate_kwargs.pop('top_p', None)
        temperature = generate_kwargs.pop('temperature', None)
        top_k = generate_kwargs.pop('top_k', None)
        eos_reached = paddle.to_tensor([False] * batch_size)

        # generation loop
        from tqdm.auto import tqdm
        for generated_tokens in tqdm(range(max_new_tokens)):
            # prepare inputs
            # warning: batched input should use left padding
            angle_mask = paddle.to_tensor(batch_with_angle), paddle.full([batch_size], input_embeds.shape[1],
                                                                         dtype='int64')
            inputs_for_generation = {
                'input_embeds': input_embeds,
                'attention_mask': attention_mask,
                'angle_mask': angle_mask,
                'angle_idx': batch_with_angle,
                'past_key_values': past_key_values,
                'use_cache': True
            }

            # do next-token prediction
            outputs = self(**inputs_for_generation)
            next_token_logits = outputs.language_model_outputs.logits[:, -1, :]

            # TODO: add support for temp, top_k, top_p
            if do_sample:
                # temperature
                if temperature is not None and temperature > 0:
                    next_token_logits = next_token_logits / temperature

                next_token_probs = paddle.nn.functional.softmax(next_token_logits, dtype='float32')
                # top_k
                if top_k is not None and top_k >= 1:
                    # only keep the first k largest probs, others are set to zero
                    top_k_probs, top_k_indices = paddle.topk(next_token_probs, top_k, axis=-1)
                    batch_indices = paddle.arange(batch_size).unsqueeze(1)
                    top_k_mask = paddle.zeros_like(next_token_probs, dtype='bool')
                    top_k_mask[batch_indices, top_k_indices] = True
                    next_token_probs[~top_k_mask] = 0
                # top_p
                if top_p is not None and top_p > 0:
                    sorted_indices = paddle.argsort(next_token_probs, descending=True, axis=-1)
                    batch_indices = paddle.arange(batch_size).unsqueeze(1)
                    sorted_probs = next_token_probs[batch_indices, sorted_indices]
                    cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)
                    top_p_mask = (cumulative_probs <= top_p)
                    # in case the largest prob alone is greater than top_p
                    top_p_mask[..., 0] = True
                    top_p_indices = paddle.argsort(sorted_indices, axis=-1)
                    batch_indices = paddle.arange(next_token_probs.shape[0]).unsqueeze(1)
                    top_p_mask = top_p_mask[batch_indices, top_p_indices]
                    next_token_probs[~top_p_mask] = 0

                next_tokens = paddle.multinomial(next_token_probs, num_samples=1, replacement=False).squeeze(-1)
            else:
                next_tokens = paddle.argmax(next_token_logits, axis=-1)

            input_ids = paddle.concat([input_ids, next_tokens[:, None]], axis=1)

            # stopping criteria
            is_eos = paddle.nonzero(next_tokens == eos_id, as_tuple=False)
            if len(is_eos) > 0:
                eos_reached[is_eos] = True
            if eos_reached.all():
                break

            # update angle_mask: only contain those with [NUM] token
            # batch_with_angle = [idx for idx in range(batch_size) if num_id in input_ids[idx]]
            # this should only be true at most at the first round of generation
            # if len(batch_with_angle) < len(angle_mask[0]) and len(angle_mask[0]) == batch_size:
            #     if len(batch_with_angle) > 0:
            #         outputs.number_decoder_outputs.pred_num = outputs.number_decoder_outputs.pred_num[batch_with_angle]
            # angle_mask = paddle.to_tensor(batch_with_angle), paddle.zeros(len(batch_with_angle), dtype='int64')

            if len(batch_with_angle) > 0:
                pred_num = outputs.number_decoder_outputs.pred_num
                pred_angles.append(pred_num)

            past_key_values = outputs.language_model_outputs.past_key_values

            # update inputs/outputs
            # text_tokens_[text_tokens_ < 0] = 1  # Not used
            next_embeds = self.language_model.get_input_embeddings()(next_tokens)
            if (next_tokens == num_id).any():
                # 使用 paddle.where 来选择特定的嵌入并添加 residue_embedding
                next_embeds[next_tokens == num_id] = next_embeds[next_tokens == num_id] + self.residue_embedding(
                    pred_num[next_tokens == num_id])
            input_embeds = next_embeds[:, None, :]
            # input_embeds = paddle.concat([input_embeds, next_embeds[:,None,:]], axis=1)
            next_token_attention_mask = paddle.ones((batch_size, 1), dtype='float32')
            attention_mask = paddle.concat([attention_mask, next_token_attention_mask.astype('int64')], axis=1)

        if len(pred_angles) > 0:
            pred_angles = paddle.concat([pred_num.unsqueeze(1) for pred_num in pred_angles], axis=1)
            pred_dtype = pred_angles.dtype
            # 将张量转换为 float64 类型
            pred_angles = pred_angles.astype('float64')
            pred_angles = (pred_angles + math.pi) % (2 * math.pi) - math.pi
            pred_angles = pred_angles.astype(pred_dtype)

        return HelixProtXGenerationOutput(input_ids=input_ids, pred_angles=pred_angles)

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            base_model_path: Optional[Union[str, os.PathLike]] = None,
            convert_from_torch=False,
            *model_args,
            **kwargs
    ):
        """
        `pretrained_model_name_or_path`: path that contains 3 sub folders ['structure_encoder', 'abstractor', 'language_model'], and a config file
        `base_model_path`: path of base LLM
        """
        # get config
        config = kwargs.pop('config', None)
        dtype = kwargs.pop('dtype', None)
        use_auth_token = kwargs.pop('use_auth_token', None)
        local_files_only = kwargs.pop('local_files_only', True)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")
        from_auto_class = kwargs.pop("_from_auto", False)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        _fast_init = kwargs.pop("_fast_init", True)
        # use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)

        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                return_unused_kwargs=True,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # instantiate model
        init_contexts = [no_init_weights(_enable=_fast_init)]
        init_contexts.append(paddle.LazyGuard())
        with ContextManagers(init_contexts):
            model = cls(config, *model_args, **model_kwargs)

        # load model from pretrained
        model.structure_abstractor = Abstractor.from_pretrained(
            os.path.join(pretrained_model_name_or_path, SAVE_SUBDIR['structure_abstractor']),
            text_hidden_size=config.text_config.hidden_size,
            convert_from_torch=convert_from_torch, dtype=dtype
        )
        model.sequence_abstractor = Abstractor.from_pretrained(
            os.path.join(pretrained_model_name_or_path, SAVE_SUBDIR['sequence_abstractor']),
            text_hidden_size=config.text_config.hidden_size,
            convert_from_torch=convert_from_torch, dtype=dtype
        )
        model.residue_embedding = ResidueEmbedding.from_pretrained(
            os.path.join(pretrained_model_name_or_path, SAVE_SUBDIR['residue_embedding']),
            convert_from_torch=convert_from_torch, dtype=dtype
        )
        model.number_decoder = NumberDecoder.from_pretrained(
            os.path.join(pretrained_model_name_or_path, SAVE_SUBDIR['number_decoder']),
            convert_from_torch=convert_from_torch, dtype=dtype
        )

        if base_model_path is None:
            # base model saved in `model_path/language_model/base`
            base_model_path = os.path.join(pretrained_model_name_or_path, SAVE_SUBDIR['language_model'], "base")
            has_base = os.path.exists(os.path.join(base_model_path, "config.json"))
            assert has_base, f"`base_model_path` not provided, and no base model was found in {pretrained_model_name_or_path}"

        # load base model
        model.language_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, convert_from_torch=convert_from_torch, dtype=dtype
        )
        model.eval()
        if model.can_generate():
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    _from_auto=from_auto_class,
                    _from_pipeline=from_pipeline,
                    **kwargs,
                )
            except (OSError, TypeError):
                # logger.info("Generation config file not found, using a generation config created from the model config.")
                pass
        return model

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            is_main_process: bool = True,
            state_dict: Optional[dict] = None,
            save_function: Callable = paddle.save,
            push_to_hub: bool = False,
            max_shard_size: Union[int, str] = "10GB",
            safe_serialization: bool = False,
            variant: Optional[str] = None,
            save_base_model: bool = True,
            **kwargs,
    ):
        os.makedirs(save_directory, exist_ok=True)
        # save config file in root
        # Only save the model itself if we are using distributed training
        model_to_save = unwrap_model(self)

        # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.torch_dtype = str(dtype).split(".")[1]

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Save the config
        if is_main_process:
            model_to_save.config.save_pretrained(save_directory)
            if self.can_generate():
                model_to_save.generation_config.save_pretrained(save_directory)

        # save model in subfolders
        self.structure_abstractor.save_pretrained(
            save_dir=os.path.join(save_directory, SAVE_SUBDIR['structure_abstractor']),
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            **kwargs,
        )
        self.sequence_abstractor.save_pretrained(
            save_dir=os.path.join(save_directory, SAVE_SUBDIR['sequence_abstractor']),
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            **kwargs,
        )
        self.residue_embedding.save_pretrained(
            save_dir=os.path.join(save_directory, SAVE_SUBDIR['residue_embedding']),
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            **kwargs,
        )
        self.number_decoder.save_pretrained(
            save_dir=os.path.join(save_directory, SAVE_SUBDIR['number_decoder']),
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            **kwargs,
        )

        lm_save_dir = os.path.join(save_directory, SAVE_SUBDIR['language_model'])
        base_model_path = os.path.join(lm_save_dir, "base")
        lora_path = os.path.join(lm_save_dir, "lora")

        # by default, does not save base LLM
        if save_base_model:
            self.language_model.save_pretrained(
                save_dir=base_model_path,
                is_main_process=is_main_process,
                state_dict=state_dict,
                save_function=save_function,
                max_shard_size=max_shard_size,
                safe_serialization=safe_serialization,
                variant=variant,
                **kwargs,
            )