"""
Simplified from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py.
"""

import math
import os
import numpy as np
import torch as th
import inspect
import torch

import gaussian_diffusion
from Dataloader import IntpDataset
from torch.nn.utils.rnn import pad_sequence
import vae as v
import Transformer as tr
import unet
from model_create import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)
import gaussian_diffusion as diffusion
import Dataloader as dl


settings = {
    'origin_path': './Datasets/Original_res250/',
    'call_path': './Datasets/Trans_Call_res250/',
    'debug': False,
    'bp': False,

    'batch': 256,
    'epoch': 1000,
    'lr': 1e-5,

    'word_col': [0, 1, 2],
    'pos_col': [3, 4, 5, 6, 7, 8, 9, 10, 11],
    'embedding_dim': 512,
    'feedforward_dim': 2048,
    'output_dim': 1,
    'num_head': 8,
    'num_layers': 6,
    'classifier_scale': 0.1
}
PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
device = torch.device("cuda")
model_vae = v.VAE(input_dim=12, h_dim=1024, z_dim=20).to(device)
dataset_train = IntpDataset(origin_path=settings['origin_path'], call_path=settings['call_path'], call_name='test', debug=settings['debug'])
dataloader_tr = torch.utils.data.DataLoader(dataset_train, batch_size=settings['batch'], shuffle=True, collate_fn=dl.collate_fn, num_workers=4, prefetch_factor=4)
model = tr.Transformer(
        word_col=settings['word_col'], pos_col=settings['pos_col'], embedding_dim=settings['embedding_dim'],
        feedforward_dim=settings['feedforward_dim'], output_dim=settings['output_dim'],
        num_head=settings['num_head'], num_layers=settings['num_layers'],
        dropout=0.1, activation="gelu", device=device
    )



model_diffusion = gaussian_diffusion.GaussianDiffusion(
    noise_schedule = "squaredcos_cap_v2", steps = 1000
)

model_unet = unet.UNetModel(
    in_channels = 12,
    model_channels = 3072,  # 256*12
    out_channels = 24,
    num_res_blocks=3,
    attention_resolutions="32,16,8",
    dropout=0.1,
    num_classes=None,
)

# for inputs_ex, lengths_ex, targets_ex in dataloader_tr:
# for inputs_ex, lengths_ex, targets_ex in dataloader_tr:
#     output_vae = model_vae(inputs_ex.to(device))
#     output_ex = model(inputs_ex.to(device), lengths_ex.to(device))

def model_fn(x_t, ts):
    half = x_t[: len(x_t) // 2]
    combined = th.cat([half, half], dim=0)
    model_out = model_unet(combined, ts)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + 3 * (cond_eps - uncond_eps)
    eps = th.cat([half_eps, half_eps], dim=0)
    return th.cat([eps, rest], dim=1)

batch_size = 256
full_batch_size = batch_size * 2


def cond_fn(self, x, t):
    B, C = x.shape[:2]
    assert t.shape == (B,)
    y = model(x, x.shape[0])
    y_reshaped = y.unsqueeze(1).unsqueeze(2).repeat(1, x.shape[1], x.shape[2])
    assert y_reshaped.shape == x.shape
    assert th.equal(y_reshaped[:, :, 0], y.repeat(1, x.shape[1]))
    # 问题：如何计算p(x|y)?
    # p_xt_var = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
    # p_xt_mean = _extract_into_tensor(self.posterior_mean_coef2+self.posterior_mean_coef1, t, x.shape) * x

samples = diffusion.GaussianDiffusion.p_sample_loop(
    model_fn,
    (full_batch_size, 12, 250, 250),
    device=device,
    clip_denoised=True,
    progress=True,
    model_kwargs=None,
    cond_fn=None,  # cond_fn(output_ex, 1000)
)[:batch_size]




