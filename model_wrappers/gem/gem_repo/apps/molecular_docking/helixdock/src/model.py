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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
transformer network
"""

from multiprocessing import reduction
import os
from copy import deepcopy
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
import numpy as np

from .basic_block import MLP, atom_pos_to_pair_dist, ligand_protein_pos_to_pair_dist
from .modules_dock import HelixDock
from .litegem_modules import LiteGEMDock

# added2
from .diffusion import GaussianDiffusionTrainer, DDIMSampler
import pdb

class LigandPosPredHead(nn.Layer):
    """
    tbd
    """
    def __init__(self, model_config, encoder_config):
        super().__init__()
        self.model_config = model_config

        self.label_mean = paddle.to_tensor(self.model_config.label_mean)
        self.label_std = paddle.to_tensor(self.model_config.label_std)

        loss_dict = {
            'l1loss': nn.L1Loss(reduction="none"),
            'huber_loss_5': nn.SmoothL1Loss(reduction="none", delta=5.0),
            'huber_loss_2': nn.SmoothL1Loss(reduction="none", delta=2.0),
            'rmsd': nn.MSELoss(reduction="none")
        }
        self.criterion = loss_dict[self.model_config.loss_type]
        self.use_diffusion = model_config.get('use_diffusion', False)
        if self.use_diffusion:
            self.mean_type = model_config.mean_type

    def _get_scaled_label(self, x):
        return (x - self.label_mean) / (self.label_std + 1e-5)

    def _get_unscaled_pred(self, x):
        return x * (self.label_std + 1e-5) + self.label_mean
    
    def _get_rmsd_loss(self, pred, label, mask):
        """
        tbd
        """
        rmsd_loss = self.criterion(pred, label).sum(-1) * mask # (B, N)
        rmsd_loss = paddle.sqrt(rmsd_loss.sum(-1) / mask.sum(-1)).mean()
        return rmsd_loss
    
    def forward(self, batch, encoder_results):
        """tbd"""
        if self.use_diffusion and self.mean_type == 'epsilon':
            eps_theta_list = encoder_results['eps_theta_list']      # (B, N, 3)
            noise = encoder_results['noise']      # (B, N, 3)
            ligand_atom_mask = batch['ligand_atom_mask']

            loss_list = []
            for eps_theta in eps_theta_list:
                # cur_loss = F.mse_loss(eps_theta, noise, reduction='none')   # (B, N, 3)
                if self.model_config.loss_type == 'rmsd':
                    cur_loss = self._get_rmsd_loss(eps_theta, noise, ligand_atom_mask)
                else:
                    cur_loss = self.criterion(eps_theta, noise) * ligand_atom_mask.unsqueeze([-1])
                    if 'valid_mask' in batch:
                        mask = batch['valid_mask'].unsqueeze([-1, -1])
                        cur_loss = (cur_loss * mask).sum() / ((ligand_atom_mask.unsqueeze([-1]) * mask).sum() + 1e-5 )
                    else:
                        cur_loss = cur_loss.sum() / ligand_atom_mask.sum()
                # cur_loss = cur_loss.mean()  # (1)
                loss_list.append(cur_loss)
            if len(loss_list) > 1:
                loss = 0.5 * paddle.mean(paddle.stack(loss_list[:-1])) + 0.5 * loss_list[-1]
            else:
                loss = loss_list[-1]
            loss *= self.model_config.loss_scale
            results = {
                'loss': loss,
            }
            return results
        else:
            ligand_pred_pos_list = encoder_results['ligand_pred_pos_list']      # [(B, N, 3)]
            scaled_label_pos = self._get_scaled_label(batch['ligand_atom_pos'])      # (B, N, 3)
            ligand_atom_mask = batch['ligand_atom_mask']

            loss_list = []
            i = 0 
            for pred_pos in ligand_pred_pos_list:
                if self.use_diffusion:
                    mask = batch['ligand_atom_mask'].sum(axis=-1).unsqueeze(-1)
                    ligand_zero_center = paddle.sum(batch['ligand_zero_atom_pos'], axis=1) / mask #(B, 3)
                    ligand_zero_center_pos = ligand_zero_center.unsqueeze(1)
                    pred_pos += ligand_zero_center_pos
                if self.model_config.loss_type == 'rmsd':
                    cur_loss = self._get_rmsd_loss(pred_pos, scaled_label_pos, ligand_atom_mask)
                else:
                    cur_loss = self.criterion(pred_pos, scaled_label_pos) * ligand_atom_mask.unsqueeze([-1])
                    if 'valid_mask' in batch:
                        mask = batch['valid_mask'].unsqueeze([-1, -1])
                        cur_loss = (cur_loss * mask).sum() / ((ligand_atom_mask.unsqueeze([-1]) * mask).sum() + 1e-5 )
                    else:
                        cur_loss = cur_loss.sum() / ligand_atom_mask.sum()
                i += 1
                loss_list.append(cur_loss)
            if len(loss_list) > 1:
                loss = 0.5 * paddle.mean(paddle.stack(loss_list[:-1])) + 0.5 * loss_list[-1]
            else:
                loss = loss_list[-1]
            all_pred_pos = [self._get_unscaled_pred(pos) for pos in ligand_pred_pos_list]
            final_pred_pos = all_pred_pos[-1]
            

        loss *= self.model_config.loss_scale
        results = {
            'all_pred_pos': all_pred_pos,
            'final_pred_pos': final_pred_pos,
            'loss': loss,
        }
        return results


class HelixDockPredictor(nn.Layer):
    """
    tbd
    """
    def __init__(self, model_config, encoder_config):
        super().__init__()
        self.model_config = deepcopy(model_config.model)
        self.encoder_config = deepcopy(encoder_config)
        
        if self.model_config.encoder_type == 'HelixDock':
            self.encoder = HelixDock(self.encoder_config)
        elif self.model_config.encoder_type == 'LiteGEM':
            self.encoder = LiteGEMDock(self.encoder_config)     
        else:
            raise ValueError(self.model_config.encoder_type)

        head_dict = {
            "ligand_atom_pos_head": LigandPosPredHead
        }

        self.use_diffusion = self.model_config.diffusion_params.get('in_use', False)
        if self.use_diffusion:
            self.diffusion_params = self.model_config.diffusion_params
            self.trainer = GaussianDiffusionTrainer(self.diffusion_params)
        print("[use_diffusion]", self.use_diffusion)

        self.heads = nn.LayerDict()
        for name in self.model_config.heads:
            model_in_use = self.model_config.heads[name].get('in_use', True)
            if model_in_use:
                self.heads[name] = head_dict[name](
                        self.model_config.heads[name], self.encoder_config)

    def _get_scaled_label(self, x):
        return (x - self.diffusion_params.label_mean) / (self.diffusion_params.label_std + 1e-5)

    def forward(self, batch):
        """tbd"""
        ligand_cur_pos_0 = self._get_scaled_label(batch['ligand_atom_pos'])      # (B, N, 3)
        mask = batch['ligand_atom_mask'].sum(axis=-1).unsqueeze(-1) 
        ligand_zero_center = paddle.sum(batch['ligand_zero_atom_pos'], axis=1) / mask#(B, 3)
        ligand_zero_center_pos = ligand_zero_center.unsqueeze(1)
        ligand_cur_pos_t, time_step = self.trainer(ligand_cur_pos_0 - ligand_zero_center_pos)
        batch['ligand_cur_pos'] = ligand_cur_pos_t
        batch['time_step'] = time_step
        # batch['noise'] = noise
        batch['protein_atom_pos_zero'] = batch['protein_atom_pos'] - ligand_zero_center_pos

        encoder_results = self.encoder(batch)
        results = {}
        total_loss = 0
        for name in self.heads:
            results[name] = self.heads[name](batch, encoder_results)
            if len(results[name]) > 0:
                total_loss += results[name]['loss']

        results['loss'] = total_loss
        return results
