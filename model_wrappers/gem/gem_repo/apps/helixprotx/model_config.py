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

"""model config"""
import paddle
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import copy
from paddlenlp.transformers.model_outputs import ModelOutput, CausalLMOutputWithPast
from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.transformers import AutoConfig, LlamaConfig


class StructureEncoderConfig(PretrainedConfig):
    pass


class SequenceEncoderConfig(PretrainedConfig):
    pass


class AbstractorConfig(PretrainedConfig):
    model_type = "abstractor"

    def __init__(
        self,
        num_tokens: int = 64,
        encoder_hidden_size: int = 128,
        hidden_size: int = 256,
        intermediate_size: int = 1024,
        num_hidden_layers: int = 1,
        layer_norm_eps: float = 1e-5,
        num_attention_heads: int = 8,
        attention_probs_dropout_prob: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.encoder_hidden_size = encoder_hidden_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm_eps = layer_norm_eps
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob


class NumberDecoderConfig(PretrainedConfig):
    model_type = "number_decoder"

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 1024,
        output_size: int = 6,
        output_scaling: Optional[float] = 1.0,
        layer_norm_eps: Optional[float] = 1e-5,
        loss_scaling: Optional[float] = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.output_size = output_size
        self.output_scaling = output_scaling
        self.layer_norm_eps = layer_norm_eps
        self.loss_scaling = loss_scaling


class ResidueEmbeddingConfig(PretrainedConfig):
    model_type = "residue_embedding"

    def __init__(
        self,
        num_angles: int = 6,
        num_embeddings: int = 64,
        embedding_dim: int = 128,
        num_range: Tuple[float] = [-math.pi, math.pi],
        bias: bool = False,
        intermediate_size: int = 512,
        layer_norm_eps: float = 1e-5,
        output_hidden_size: int = 4096,
        num_attention_heads: int = 8,
        attention_probs_dropout_prob: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # NumberEmbedding config
        self.num_angles = num_angles
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_range = num_range
        self.bias = bias

        # ResidueEmbeddingAttention config
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.output_hidden_size = output_hidden_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob


class HelixProtXConfig(PretrainedConfig):
    model_type = "helixprotx"
    is_composition = True

    def __init__(
        self,
        structure_abstractor_config: AbstractorConfig = None,
        sequence_abstractor_config: AbstractorConfig = None,
        residue_embedding_config: ResidueEmbeddingConfig = None,
        number_decoder_config: NumberDecoderConfig = None,
        text_config: PretrainedConfig = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if structure_abstractor_config is None:
            structure_abstractor_config = AbstractorConfig()
            # logger.info("structure_abstractor_config is None. Using default.")
        elif isinstance(structure_abstractor_config, dict):
            structure_abstractor_config = AbstractorConfig(**structure_abstractor_config)

        if sequence_abstractor_config is None:
            sequence_abstractor_config = AbstractorConfig()
            # logger.info("structure_abstractor_config is None. Using default.")
        elif isinstance(sequence_abstractor_config, dict):
            sequence_abstractor_config = AbstractorConfig(**sequence_abstractor_config)

        if residue_embedding_config is None:
            residue_embedding_config = ResidueEmbeddingConfig()
            # logger.info("residue_embedding_config is None. Using default.")
        elif isinstance(residue_embedding_config, dict):
            residue_embedding_config = ResidueEmbeddingConfig(**residue_embedding_config)

        if number_decoder_config is None:
            number_decoder_config = NumberDecoderConfig()
            # logger.info("number_decoder_config is None. Using default.")
        elif isinstance(number_decoder_config, dict):
            number_decoder_config = NumberDecoderConfig(**number_decoder_config)

        if text_config is None:
            text_config = LlamaConfig()
            setattr(text_config, 'architectures', ['LlamaForCausalLM']) # temp fix
            # logger.info("text_config is None. Using default.")
        elif isinstance(text_config, dict):
            # text_config = AutoConfig(**text_config)
            text_config = LlamaConfig(**text_config) # temp fix
            setattr(text_config, 'architectures', ['LlamaForCausalLM']) # temp fix

        self.structure_abstractor_config = structure_abstractor_config
        self.sequence_abstractor_config = sequence_abstractor_config
        self.residue_embedding_config = residue_embedding_config
        self.number_decoder_config = number_decoder_config
        self.text_config = text_config
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.is_encoder_decoder = self.text_config.is_encoder_decoder

        self.initializer_factor = 1.0
        self.initializer_range = 0.02

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["structure_abstractor_config"] = self.structure_abstractor_config.to_dict()
        output["sequence_abstractor_config"] = self.sequence_abstractor_config.to_dict()
        output["residue_embedding_config"] = self.residue_embedding_config.to_dict()
        output["number_decoder_config"] = self.number_decoder_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


# register config
# for config_cls in (AbstractorConfig, NumberDecoderConfig, ResidueEmbeddingConfig, HelixProtXConfig):
#     AutoConfig.register(config_cls.model_type, config_cls)


@dataclass
class NumberDecoderOutput(ModelOutput):
    pred_num: paddle.Tensor = None
    loss: Optional[paddle.Tensor] = None
    range_loss: Optional[paddle.Tensor] = None
    rad_loss: Optional[paddle.Tensor] = None
    rad_loss_by_angle_type: Optional[paddle.Tensor] = None


@dataclass
class HelixProtXGenerationOutput(ModelOutput):
    input_ids: Optional[paddle.Tensor] = None
    pred_angles: Optional[paddle.Tensor] = None


@dataclass
class HelixProtXOutput(ModelOutput):
    language_model_outputs: Optional[CausalLMOutputWithPast] = None
    number_decoder_outputs: Optional[NumberDecoderOutput] = None


@dataclass
class EncoderOutput(ModelOutput):
    encoder_hidden_states: Optional[paddle.Tensor] = None
    encoder_attention_mask: Optional[paddle.Tensor] = None

