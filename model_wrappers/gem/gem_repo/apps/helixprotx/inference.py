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
inference demo
make sure you run `build_model.py` first to create a `HelixProtX` checkpoint
"""
import os
from model import HelixProtX
from paddlenlp.transformers import AutoTokenizer
import paddle


def get_dummy_inputs(config, mode):
    assert mode in ['forward', 'generate']
    from model_config import EncoderOutput

    # random inputs
    batch_size = 6
    max_input_len = 200
    input_ids = paddle.zeros((batch_size, max_input_len))
    ignore_id = -100
    labels = input_ids.clone()
    labels[:, max_input_len // 2] = ignore_id
    attention_mask = paddle.ones_like(input_ids)

    max_struc_len = 111
    max_seq_len = 112
    struc_idx = [0, 1]
    seq_idx = [2, 3]
    angle_idx = [3, 5]

    structure_encoder_outputs = EncoderOutput(
        encoder_hidden_states=paddle.randn(
            (len(struc_idx), max_struc_len, config.structure_abstractor_config.encoder_hidden_size)),
        encoder_attention_mask=paddle.ones((len(struc_idx), max_struc_len))
    )
    sequence_encoder_outputs = EncoderOutput(
        encoder_hidden_states=paddle.randn(
            (len(seq_idx), max_seq_len, config.sequence_abstractor_config.encoder_hidden_size)),
        encoder_attention_mask=paddle.ones((len(seq_idx), max_seq_len))
    )

    num_id = 123
    input_ids[angle_idx, -20:] = num_id
    angle_mask = (input_ids == num_id)

    if mode == 'forward':
        batched_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'structure_encoder_outputs': structure_encoder_outputs,
            'sequence_encoder_outputs': sequence_encoder_outputs,
            'angle_labels': angle_labels,
            'angle_mask': angle_mask,
            'label_ids': labels,
            'struc_idx': struc_idx,
            'seq_idx': seq_idx,
            'angle_idx': angle_idx,
        }
    elif mode == 'generate':
        batched_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'structure_encoder_outputs': structure_encoder_outputs,
            'struc_idx': struc_idx,
            'sequence_encoder_outputs': sequence_encoder_outputs,
            'seq_idx': seq_idx
        }
    return batched_inputs


# load model and tokenizer
model_path = 'test_model'  # 'test_model'
model = HelixProtX.from_pretrained(model_path, dtype='float16')
# tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, 'tokenizer'))
# breakpoint()
# run forward pass
model_inputs = get_dummy_inputs(model.config, mode='forward')
with paddle.no_grad():
    output = model(**model_inputs)

# print loss
print(f"lm_loss: {output.language_model_outputs.loss.item()}")
if output.number_decoder_outputs is not None:
    print(f"rad_loss: {output.number_decoder_outputs.rad_loss_by_angle_type.tolist()}")

# run generation
generation_inputs = get_dummy_inputs(model.config, mode='generate')
output = model.generate(**generation_inputs)

# show generation results
gen_results = []
for batch_idx in range(len(output.input_ids.shape[0])):
    output_sentence = tokenizer.decode(output.input_ids[batch_idx].tolist())

# save angles as pdb file
if output.pred_angles is not None:
    pdb_dir = 'generated_pdb'
    for batch_idx, angles in zip(generation_inputs, output.pred_angles):
        save_file = os.path.join(pdb_dir, f'{batch_idx}.pdb')
        angles_to_pdb_file(angles_pred.cpu().detach(), save_file)
