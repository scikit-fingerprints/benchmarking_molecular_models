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

"""
add [NUM] token and build multimodal model
"""
import os
from model import HelixProtX
from model_config import HelixProtXConfig, ResidueEmbeddingConfig, NumberDecoderConfig
from paddlenlp.transformers import LlamaConfig, AutoModelForCausalLM, AutoTokenizer


# init text config
text_config = LlamaConfig()
setattr(text_config, 'architectures', ['LlamaForCausalLM'])

# init residue embedding config
residue_embedding_config = ResidueEmbeddingConfig(output_hidden_size=text_config.hidden_size)

# init number decoder config
number_decoder_config = NumberDecoderConfig(hidden_size=text_config.hidden_size)

# init helixprotx config (with default abstractor config)
config = HelixProtXConfig(
    text_config=text_config,
    residue_embedding_config=residue_embedding_config,
    number_decoder_config=number_decoder_config
)

# init model
model = HelixProtX(config)

# load pretrained llm and tokenizer
model.language_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat', dtype='bfloat16')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat')

# add [NUM] token (or use reserved tokens)
tokenizer.add_tokens('[NUM]')
if len(tokenizer) > model.language_model.get_input_embeddings().weight.shape[0]:
    model.language_model.resize_token_embeddings(len(tokenizer))
model.language_model.config.vocab_size = len(tokenizer)
model.config.text_config.vocab_size = len(tokenizer)

# save as HelixProtX checkpoint
output_dir = 'test_model'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(os.path.join(output_dir, 'tokenizer'))