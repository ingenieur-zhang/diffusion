import torch.nn as nn
import os, glob, inspect, time, math, torch, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
import Datasets.Dataloader_PEGNN as dl
import torch.optim as optim
from torch.nn import functional as F
import mpl_scatter_density
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import Trans_scrpit.myconfig as myconfig
from datetime import datetime
import json
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LambdaLR

# def save_square_img(contents, xlabel, ylabel, savename, title):
#     # "Viridis-like" colormap with white background
#     white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
#         (0, '#ffffff'),
#         (1e-20, '#440053'),
#         (0.2, '#404388'),
#         (0.4, '#2a788e'),
#         (0.6, '#21a784'),
#         (0.8, '#78d151'),
#         (1, '#fde624'),
#     ], N=256)
    
#     plt.clf()
#     plt.rcParams['font.size'] = 15
    
#     max_value = max(contents[0].max(), contents[1].max())
    
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
#     fig.set_size_inches(7, 6)
#     ax.set_position([0, 0, 0.8, 1])
    
#     density = ax.scatter_density(contents[0], contents[1], cmap=white_viridis)
#     fig.colorbar(density, label='Number of points')

#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
#     fig.tight_layout(pad=1, w_pad=1, h_pad=1)
    
    
#     ax.set_xlim([0, max_value])
#     ax.set_ylim([0, max_value])
#     ax.plot([0, max_value], [0, max_value], color='k')
#     fig.savefig("%s.png" %(savename))
#     plt.close(fig)
# coffer_slot = './coffer_slot'
# settings = {
#         'origin_path': '../Datasets/Dataset_res250/',
#         'debug': False,
#         'bp': False,
#         'debug_stage': 1,
#         'agent_dir': 'coffer_slot/',
        
#         'batch': 256,
#         'accumulation_steps': 1,
#         'epoch': 20000,
#         'trans_lr': 1e-6,
#         'nn_lr': 1e-4,
#         'es_mindelta': 1.0,
#         'es_endure': 30,

#         'word_col': [0, 1],
#         'pos_col': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
#         'q_col': 3,

#         'embedding_dim': 32,
#         'feedforward_dim': 4 * 32,
#         'num_head': 2,
#         'num_layers': 2,
#         'output_hidden_layers': 2,
#         'vi_dim': 16,
#         'dropout': 0,

#         'fold': 0,
#         'holdout': 0,
#         'lowest_rank': 1,
        
#         'T': 4 ,
#         'ddim_step':2,
#         'cdim': 16,
#         'beta_1': 1e-4,
#         'beta_T': 0.02,
#         'channel': 32,
#         'tdim': 16,
#         'nheads': 8, 
#         'k': 20,
#         'emb_hidden_dim': 256,
#         'emb_dim': 16,
#         'input_emb': 16,
        
#     }

# def vae_loss(recon_x, x, mu, logvar):
#     reconstruction_loss = nn.MSELoss()(recon_x, x)
#     kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     print('reconstruction_loss: ', reconstruction_loss)
#     print('kl_divergence: ', kl_divergence)
#     return reconstruction_loss + kl_divergence

# class NewGELU(nn.Module):
#     def forward(self, x):
#         return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

# 构建一个简单的全连接神经网络
# class VAE(nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super(VAE, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 64)
#         self.fc_mu = nn.Linear(64, latent_dim)
#         self.fc_logvar = nn.Linear(64, latent_dim)
#         self.fc2 = nn.Linear(latent_dim, 64)
#         self.fc3 = nn.Linear(64, input_dim)

#     def encode(self, x):
#         x = torch.relu(self.fc1(x))
#         mu = self.fc_mu(x)
#         logvar = self.fc_logvar(x)
#         return mu, logvar

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         z = torch.relu(self.fc2(z))
#         x = self.fc3(z)
#         return x

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         x_recon = self.decode(z)
#         return x_recon, mu, logvar

# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch import optim
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.datasets import MNIST
# import matplotlib.pyplot as plt

# latent_dim = 512
# input_dim = 12
# inter_dim = 1024

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)
        self.fc3 = nn.Linear(64, input_dim)

    def encode(self, x):
        x = torch.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = torch.relu(self.fc2(z))
        x = self.fc3(z)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# if __name__ == '__main__':
#     batch_size = 2
#     model = VAE(batch_size)
#     x = torch.randn(batch_size, 20, 1)
#     # t = torch.randint(1000, (batch_size,))
#     y, mu, var = model(x)

#     print(y.shape)


# 创建混合高斯分布的 VAE
# vae = VAE(latent_dim=2, num_mixtures=5)

# # 构建模型和损失函数
# model = VAE(input_dim=1, latent_dim=256)
# criterion = nn.MSELoss()  # 均方误差损失函数
# optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器


# dataset_train = dl.IntpDataset(settings=settings, mask_distance=-1, call_name='train')
# dataloader_tr = torch.utils.data.DataLoader(dataset_train, batch_size=settings['batch'], shuffle=True, collate_fn=dl.collate_fn, num_workers=10, prefetch_factor=32, drop_last=True)
# print('train data loaded')

# test_dataloaders = []
# test_trans_dataloaders = []

# dataset_test = dl.IntpDataset(settings=settings, mask_distance=20, call_name='test')
# dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=settings['batch'], shuffle=False, collate_fn=dl.collate_fn, num_workers=10, prefetch_factor=32, drop_last=True)
# print('test data loaded')

# with open(settings['origin_path'] + f'Folds_Info/norm_{0}.info', 'rb') as f:
#     dic_op_minmax, dic_op_meanstd = pickle.load(f)

# iter_counter = 0
# # 训练模型
# # model.load_state_dict(torch.load('model_weights.pth'))
# # for x_b, c_b, y_b, x_pm, q_tr, input_lenths in dataloader_tr:
# #     iter_counter += 1
# #     outputs, mu, var = model(x_pm)
# #     loss = vae_loss(outputs, y_b, mu, var)
# #     loss.backward()
# #     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.02)
# #     optimizer.step()
# #     print(f'\t Iter {iter_counter} - Loss: {loss.item()}')
# #     iter_counter += 1
# #     optimizer.zero_grad()
# #     if iter_counter % 1000 == 0:
# #         torch.save(model.state_dict(), 'model_weights_2epoch.pth')
       

# # torch.save(model.state_dict(), 'model_model_weights_2epochweights.pth')
# model.load_state_dict(torch.load('model_weights_2epoch.pth'))


# i = 0
# for x_b, c_b, y_b, x_pm, q_tr, input_lenths in dataloader_test:
#     predicted, mu, var = model(x_pm)
#     test_loss = vae_loss(predicted[:, 0:1, :], y_b[:, 0:1, :], mu, var)
#     print('test_loss:, ', test_loss)
#     output = predicted.squeeze().reshape(-1)
#     target = y_b.squeeze().reshape(-1)
#     i += 1
#     if i > 50:
#         break
        
# test_means_origin = output * dic_op_meanstd['mcpm10'][1] + dic_op_meanstd['mcpm10'][0]
# test_y_origin = target * dic_op_meanstd['mcpm10'][1] + dic_op_meanstd['mcpm10'][0]

# test_means_origin = test_means_origin.cpu().detach().numpy()
# test_y_origin = test_y_origin.cpu().detach().numpy()

# mae = mean_squared_error(test_y_origin, test_means_origin, squared=False)
# r_squared = stats.pearsonr(test_y_origin, test_means_origin)
    
# save_square_img(
#                                 contents=[test_y_origin, test_means_origin],
#                                 xlabel='targets_ex', ylabel='output_ex',
#                                 savename=os.path.join(coffer_slot, "small model"),
#                                 title=f'MSE {round(mae, 2)}'
#                             )
# # 使用模型进行预测


# # 输出模型的权重和偏置
# print("模型的权重 (fc1):", model.fc1.weight)
# print("模型的偏置 (fc1):", model.fc1.bias)
# print("模型的权重 (fc2):", model.fc2.weight)
# print("模型的偏置 (fc2):", model.fc2.bias)