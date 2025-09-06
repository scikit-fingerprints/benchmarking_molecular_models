#!/usr/bin/python3                                                                                                                                  
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
train for HelixDock
"""


import os
from os.path import join, exists, basename
import random
import argparse
import numpy as np
from glob import glob
import csv
import json
from copy import deepcopy
import ml_collections
import logging
import time

import paddle
import paddle.distributed as dist


from src.dataset import HelixDockDataset
from src.model import HelixDockPredictor
from src.featurizer import HelixDockCollateFn
from src.paddle_utils import maybe_to_tensor
from src.utils import calc_parameter_size, tree_map, set_logging_level
from src.metric import Metric
from src.config import make_updated_config, HelixDock_MODEL_CONFIG, PREDICTOR_CONFIG
from src.diffusion import DDIMSampler


def get_collate_fn(model_config, encoder_config):
    func_dict = {
        'HelixDock': HelixDockCollateFn(model_config, encoder_config),
    }
    return func_dict[model_config.model.encoder_type]


def create_model_config(args):
    model_config = make_updated_config(
            PREDICTOR_CONFIG, args.model_config)

    encoder_config_dict = {
        'HelixDock': HelixDock_MODEL_CONFIG,
    }
    encoder_config = make_updated_config(
            encoder_config_dict[model_config.model.encoder_type], 
            args.encoder_config)
    return model_config, encoder_config


@paddle.no_grad()
def evaluate(
        args, cur_step, model, 
        test_dataset, dataset_config, collate_fn,
        output_dir=None):
    """
    Define the evaluate function 
    """
    model.eval()

    data_gen = paddle.io.DataLoader(test_dataset,
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=False,
        collate_fn=collate_fn)

    metric = Metric(
            save_output=dataset_config.save_output,
            output_dir=output_dir,
            output_num=dataset_config.output_num,
            save_ligand_rmsd=True)


    diffusion_params = dataset_config.diffusion_params
    if args.distributed:
        sampler = DDIMSampler(
                model._layers.encoder, diffusion_params)
    else:
        sampler = DDIMSampler(
                model.encoder, diffusion_params)

    for batch in data_gen:
        batch = tree_map(maybe_to_tensor, batch)
        label_mean = paddle.to_tensor(dataset_config.label_mean)
        label_std = paddle.to_tensor(dataset_config.label_std)
        all_pred_pos = []
        results = model(batch)
        encoder_results = sampler(batch)
        ligand_pred_pos_list = encoder_results['ligand_pred_pos_list']    
        for pos in ligand_pred_pos_list:
            get_unscaled_pred_pos = pos * (label_std + 1e-5) + label_mean
            all_pred_pos.append(get_unscaled_pred_pos)
        final_pred_pos = all_pred_pos[-1]
        results['ligand_atom_pos_head']['all_pred_pos'] = all_pred_pos
        results['ligand_atom_pos_head']['final_pred_pos'] = final_pred_pos
        metric.add(batch, results)
    results = metric.get_result(args.distributed)
    return results


def evaluate_full(
        args, cur_step, 
        model, test_dataset_dict, 
        test_dataset_config, collate_fn):
    for name, dataset in test_dataset_dict.items():
        print(f'start evaluating {name}')
        s_time = time.time()

        results = evaluate(
                args, cur_step, model, 
                dataset, test_dataset_config[name], collate_fn,
                output_dir=f'{args.log_dir}/save_output/step{cur_step}/{name}')

        print(results)
        print(f'evaluate {name} use {time.time() - s_time} sec, data num: {len(dataset)}, batch size : {args.batch_size}')


def main(args):
    """
    Call the configuration function of the model, build the model and load data, then start training.
    model_config:
        a json file  with the hyperparameters,such as dropout rate ,learning rate,num tasks and so on;
    num_tasks:
        it means the number of task that each dataset contains, it's related to the dataset;
    """
    def _read_json(path):
        return ml_collections.ConfigDict(json.load(open(path, 'r')))

    set_logging_level(args.logging_level)

    print(f'args:\n{args}')
    dataset_config = _read_json(args.dataset_config)
    print(f'>>> dataset_config:\n{dataset_config}')
    train_config = _read_json(args.train_config)
    print(f'>>> train_config:\n{train_config}')
    model_config, encoder_config = create_model_config(args)
    print(f'>>> model_config:\n{model_config}')
    print(f'>>> encoder_config:\n{encoder_config}')

    ### init dist
    trainer_id = 0
    trainer_num = 1
    if args.distributed:
        dist.init_parallel_env()
        trainer_id = dist.get_rank()
        trainer_num = dist.get_world_size()
    paddle.seed(64)
    collate_fn = get_collate_fn(model_config, encoder_config)

    dataset_common_argv = {
        'model_config': model_config, 
        'encoder_config': encoder_config, 
        'trainer_id': trainer_id,       # data is distributed in different node
        'trainer_num': trainer_num, 
        'num_workers': args.num_workers,
    }
    test_dataset_dict = {}
    for name in dataset_config.test:
        test_dataset = HelixDockDataset(dataset_config.test[name], **dataset_common_argv)
        test_dataset_dict[name] = test_dataset
        logging.info(f"test {name} dataset num: {len(test_dataset)}")


    ### build model
    model = HelixDockPredictor(model_config, encoder_config)
    model_f = open(f'{args.log_dir}/model', 'w')
    print("model: ", model, file=model_f)
    model_f.close()
    print("parameter size:", calc_parameter_size(model.parameters()))
    if args.distributed:
        model = paddle.DataParallel(model)

    state_dict = paddle.load(args.init_model)
    model.set_state_dict(state_dict)
    print('Load state_dict from %s' % args.init_model)
    print('state param num : ', len(state_dict.keys()))

    evaluate_full(args, 0, 
            model, test_dataset_dict, 
            dataset_config.test, collate_fn)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action='store_true', default=False)
    parser.add_argument("--logging_level", type=str, default="DEBUG", 
            help="NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_step", type=int, default=20)

    parser.add_argument("--dataset_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--encoder_config", type=str)
    parser.add_argument("--train_config", type=str)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--log_dir", type=str, default="./default_log")
    args = parser.parse_args()
    
    main(args)


