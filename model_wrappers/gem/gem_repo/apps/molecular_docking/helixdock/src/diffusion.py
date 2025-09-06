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
diffusion for HelixDock
"""
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
from paddle.nn import initializer
import pdb
import random


class Swish(nn.Layer):
    """
    tbd
    """
    def forward(self, x):
        """
        tbd
        """
        sigmoid = nn.Sigmoid()
        return x * sigmoid(x)


class TimeEmbedding(nn.Layer):
    """
    tbd
    """
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = paddle.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = paddle.exp(-emb)
        pos = paddle.arange(T, dtype='float32')
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = paddle.stack([paddle.sin(emb), paddle.cos(emb)], axis=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.reshape([T, d_model])
        embedding_layer = nn.Embedding(num_embeddings=emb.shape[0], embedding_dim=emb.shape[1])
        embedding_layer.weight.set_value(emb)

        self.timembedding = nn.Sequential(
            embedding_layer,
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        """
        tbd
        """
        for module in self.sublayers():
            if isinstance(module, nn.Linear):
                # init.xavier_uniform_(module.weight)
                # init.zeros_(module.bias)
                initializer.XavierUniform(module.weight)
                initializer.XavierUniform(0, module.bias)

    def forward(self, t):
        """
        tbd
        """
        emb = self.timembedding(t)
        return emb


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    # out = paddle.gather(v, index=t, dim=0).float()
    out = paddle.gather(v, index=t)
    return out.reshape([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Layer):
    """
    tbd
    """
    def __init__(self, diffusion_params):
        beta_1 = diffusion_params['beta_1']
        beta_T = diffusion_params['beta_T']
        T = diffusion_params['T']
        super().__init__()
        self.T = T
        self.normal_mean = diffusion_params['normal_mean']
        self.normal_std = diffusion_params['normal_std']
        # tmp3
        # self.normal_std = 5

        # self.register_buffer(
        #     'betas', paddle.linspace(beta_1, beta_T, T).double())
        self.register_buffer(
            'betas', paddle.linspace(beta_1, beta_T, T))
        alphas = 1. - self.betas
        alphas_bar = paddle.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', paddle.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', paddle.sqrt(1. - alphas_bar))
        
        S = 20
        full_time_steps = list(range(T))
        sampled_time_steps = [x[0] for x in np.array_split(full_time_steps, S)]
        if not T - 1 in sampled_time_steps:
            sampled_time_steps.append(T - 1)
        self.sampled_time_steps = sampled_time_steps

    def forward(self, x_0, noise_center=0):
        """
        Algorithm 1.
        """
        t = paddle.randint(self.T, shape=(x_0.shape[0], ))
        # tmp_s
        winner = random.choice(self.sampled_time_steps)
        # winner = self.T - 1
        t = paddle.full_like(t, winner)
        # noise = paddle.randn_like(x_0)
        noise = paddle.normal(noise_center, self.normal_std, shape=x_0.shape)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        # return x_t, t, noise
        return x_t, t


class DDIMSampler(nn.Layer):
    """
    tbd
    """
    def __init__(self, model, diffusion_params):
        beta_1 = diffusion_params['beta_1']
        beta_T = diffusion_params['beta_T']
        T = diffusion_params['T']
        S = diffusion_params['S']

        eta = diffusion_params['eta']

        normal_mean = diffusion_params['normal_mean']
        normal_std = diffusion_params['normal_std']

        if 'mean_type' in diffusion_params:
            mean_type = diffusion_params['mean_type']
        else:
            mean_type = 'xstart'
        # tmp
        if 'var_type' in diffusion_params:
            var_type = diffusion_params['var_type']
        else:
            var_type = 'fixedlarge'
        assert mean_type in ['xprev', 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.S = S
        self.mean_type = mean_type
        self.var_type = var_type
        self.normal_mean = normal_mean
        self.normal_std = normal_std

        full_time_steps = list(range(T))
        sampled_time_steps = [x[0] for x in np.array_split(full_time_steps, S)]
        if not T - 1 in sampled_time_steps:
            sampled_time_steps.append(T - 1)
            self.i = 1
        else:
            self.i = 0
        self.sampled_time_steps = sampled_time_steps
        print('DDIM sampled_time_steps:', sampled_time_steps)

        self.register_buffer(
            'betas', paddle.linspace(beta_1, beta_T, T))
        ddpm_alphas = 1. - self.betas
        alphas = paddle.cumprod(ddpm_alphas, dim=0)
        sampled_time_steps_tensor = paddle.to_tensor(sampled_time_steps)
        alphas = paddle.gather(alphas, sampled_time_steps_tensor)

        alphas_prev = F.pad(alphas, [1, 0], value=1)[:-1]
        posterior_var = (1. - alphas_prev) / (1. - alphas) * (1 - alphas / alphas_prev)
        posterior_var *= eta ** 2

        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_prev', alphas_prev)
        self.register_buffer('posterior_var', posterior_var)
        self.register_buffer(
            'posterior_log_var_clipped',
            paddle.log(
                paddle.concat([posterior_var[1:2], posterior_var[1:]])))
        self.register_buffer('sigmas', paddle.sqrt(posterior_var))

    def q_mean_variance(self, x_t, eps, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, eps)
        """
        def _get_coef_t(coef_tensor):
            return extract(coef_tensor, t, x_t.shape)
        assert eps.shape == x_t.shape
        x_0 = (x_t - paddle.sqrt(1 - _get_coef_t(self.alphas)) * eps) / \
                paddle.sqrt(_get_coef_t(self.alphas))
        posterior_mean = paddle.sqrt(_get_coef_t(self.alphas_prev)) * x_0 + \
                paddle.sqrt(1 - _get_coef_t(self.alphas_prev) - _get_coef_t(self.sigmas) ** 2) * eps

        posterior_log_var_clipped = _get_coef_t(self.posterior_log_var_clipped)
        return posterior_mean, posterior_log_var_clipped

    def q_mean_variance_x0(self, x_t, x_0, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, eps)
        """
        def _get_coef_t(coef_tensor):
            return extract(coef_tensor, t, x_t.shape)
        assert x_0.shape == x_t.shape
        # x_0 = (x_t - paddle.sqrt(1 - _get_coef_t(self.alphas)) * eps) / \
        #         paddle.sqrt(_get_coef_t(self.alphas))
        eps = (x_t - paddle.sqrt(_get_coef_t(self.alphas)) * x_0) / \
                paddle.sqrt(1 - _get_coef_t(self.alphas))
        posterior_mean = paddle.sqrt(_get_coef_t(self.alphas_prev)) * x_0 + \
                paddle.sqrt(1 - _get_coef_t(self.alphas_prev) - _get_coef_t(self.sigmas) ** 2) * eps
        posterior_log_var_clipped = _get_coef_t(self.posterior_log_var_clipped)
        return posterior_mean, posterior_log_var_clipped

    def p_mean_variance(self, x_t, t, real_t, batch):
        """
        tbd
        """
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': paddle.log(paddle.concat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            # eps = self.model(x_t, real_t)
            # input of model changed
            batch['ligand_cur_pos'] = x_t
            batch['time_step'] = real_t
            results = self.model(batch)
            x_0 = results['ligand_pred_pos_list'][-1]
            model_mean, _ = self.q_mean_variance_x0(x_t, x_0, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            # eps = self.model(x_t, real_t)
            # input of model changed
            batch['ligand_cur_pos'] = x_t
            batch['time_step'] = real_t
            batch['noise'] = None
            results = self.model(batch)
            eps = results['eps_theta_list'][-1]
            model_mean, _ = self.q_mean_variance(x_t, eps, t)
        else:
            raise NotImplementedError(self.mean_type)
        return model_mean, model_log_var, results
        # tmp2
        # return model_mean, model_log_var, x_0

    def forward(self, batch):
        """
        Algorithm 2.
        """
        pro_mask = batch['protein_atom_mask'].sum(axis=-1).unsqueeze(-1)
        protein_atom_pos_center = paddle.sum(batch['protein_atom_pos'], axis=1) / pro_mask
        N = batch['ligand_atom_pos'].shape[1]
        noise_center = protein_atom_pos_center.unsqueeze(1).tile([1, N, 1])
        # tmp4
        noise_center = 0
        protein_center = protein_atom_pos_center.unsqueeze(1)
        x_T = paddle.normal(noise_center, self.normal_std, shape=batch['ligand_atom_pos'].shape)
        x_t = x_T
        ligand_pred_pos_list = []

        mask = batch['ligand_atom_mask'].sum(axis=-1).unsqueeze(-1) 
        ligand_zero_center = paddle.sum(batch['ligand_zero_atom_pos'], axis=1) / mask
        ligand_zero_center_pos = ligand_zero_center.unsqueeze(1)
        ligand_pred_pos_list.append(x_t + ligand_zero_center_pos)
        for time_step in reversed(range(self.i, self.S + self.i)):
            real_t = self.sampled_time_steps[time_step]
            t = paddle.ones([x_T.shape[0], ], dtype='int64') * time_step
            real_t = paddle.ones([x_T.shape[0], ], dtype='int64') * real_t
            mean, log_var, results = self.p_mean_variance(x_t=x_t, t=t, real_t=real_t, batch=batch)
            if time_step > self.i:
                noise = paddle.normal(noise_center, self.normal_std, shape=x_t.shape)
            else:   # time_step == 1
                noise = 0
            x_t = mean + paddle.exp(0.5 * log_var) * noise
            ligand_pred_pos_list.append(x_t + ligand_zero_center_pos)
        results['ligand_pred_pos_list'] = ligand_pred_pos_list
        return results
    
