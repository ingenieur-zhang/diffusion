import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os


# from Model_condition import UNet

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    v = v.to(device)
    out = torch.gather(v, index=t, dim=0).to(device).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T 

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, condition, targets, coords, input_lenths, tif):  # x[batch, len, 1]
        """
        Algorithm 1.
        """
        device = x_0.device
        # print(x_0.shape)
       
        # y_0 = targets.squeeze()
        # x_0 = x_0.squeeze()
        
        # sample_rate = torch.ceil(torch.abs(targets[:, 0:1, :].squeeze())).to(device).to(torch.long)
        # # print('rate before, ', sample_rate)
        # if torch.all(sample_rate == 1).item():  # x[batch, len, 1]
        #     x_filtered = x_0[0:1, :, :]
        #     y_filtered = targets[0:1, :, :]
        #     c_filtered = coords[0:1, :, :]
        #     l_filtered = input_lenths[0:1]
        #     condition_filtered = condition[0:1, :]
        #     # print('rate after, ',sample_rate)
        #     # print('y, ', y_filtered.shape)
        # else:   
        #     sample_rate_filtered = sample_rate[sample_rate != 1]
        #     y_filtered = targets[sample_rate != 1]
        #     x_filtered = x_0[sample_rate != 1]
        #     c_filtered = coords[sample_rate != 1]
        #     l_filtered = input_lenths[sample_rate != 1]
        #     condition_filtered = condition[sample_rate != 1]
           
        # print('rate after, ', sample_rate_filtered)
            # print('y, ', y_filtered.shape)
            
            # x_0 = torch.repeat_interleave(x_0, sample_rate_filtered, dim=0).unsqueeze(-1)
            # y_0 = torch.repeat_interleave(y_0, sample_rate_filtered, dim=0).unsqueeze(-1)
            # c = torch.repeat_interleave(coords, sample_rate_filtered, dim=0)
            # l = torch.repeat_interleave(input_lenths, sample_rate_filtered, dim=0)
            # condition = torch.repeat_interleave(condition, sample_rate_filtered, dim=0)


        x_filtered = x_0
        y_filtered = targets
        c_filtered = coords
        l_filtered = input_lenths
        condition_filtered = condition

        
        t = torch.randint(self.T, size=(x_filtered.shape[0],), device=x_0.device) 
        # print(y_0)
        # print('c, ', c.shape)
        # print('l, ', l.shape)
        
        # x_0 = x_0[:, :, 0:1]
        # print('y_0, ', y_0.shape)
        # print('x, ', x_0.shape)
        noise = torch.randn_like(x_filtered) 
        # print(t)
        x_t = (                                                           # x_t[batch, len, 10]
                extract(self.sqrt_alphas_bar, t, y_filtered.shape) * y_filtered +
                extract(self.sqrt_one_minus_alphas_bar, t, y_filtered.shape) * x_filtered)

        # if x_t.shape[0] == 1:
        #     eps = (y_filtered - x_t) * t / self.T
        # else:
        #     t_expanded = t.unsqueeze(1).expand( x_t.shape[0],  x_t.shape[1]).unsqueeze(-1)
            # t.view(x_t.shape[0], 1, 1).expand(-1, x_t.shape[1], -1).squeeze()
            # print((y_filtered - x_t).shape, t.shape)
            
            # eps = ((y_filtered - x_t) * t_expanded / self.T)
            # print(((y_filtered - x_t) * t).shape)
        # print(eps.shape)
        eps = noise
        output_list_full, target_list_full, coord_list_full, mask = self.model(x_t.to(device), y_filtered.to(device),
                                                                               c_filtered.to(device),
                                                                               l_filtered.to(device),
                                                                               condition_filtered.to(device),
                                                                               t.to(device), tif)
        # mask = (output_list_full != -1)
        # print(mask.shape)
        # print(noise.shape)
        # print(output_list_full.shape)
        filtered_noise = eps[:, 0:1, :][mask[:, 0:1, :]]   
        filtered_output = output_list_full[:, 0:1, :][mask[:, 0:1, :]]
        recon_y = targets[:, 0:1, :][mask[:, 0:1, :]]
        recon_x = filtered_output + x_0[:, 0:1, :][mask[:, 0:1, :]]
        loss = F.mse_loss(filtered_output, filtered_noise.to(device), reduction='none')  # 查表求出xt，
        # loss_recon = F.mse_loss(recon_x.to(device), recon_y.to(device), reduction='none')  # 查表求出xt，

        # loss = loss + loss_recon
        # recon_loss = F.mse_loss(recon_x, targets, reduction='none')
        return output_list_full, target_list_full, coord_list_full, mask, loss.mean()


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, ddim_step, eta):
        super().__init__()
        self.eta = eta
        self.model = model
        self.T = T 
        self.ddim_step = ddim_step
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.alphas_cumprod = torch.tensor(np.cumprod(alphas, axis=0))
        self.alphas_cumprod_prev = torch.tensor(np.append(1.0, self.alphas_cumprod[:-1]))
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.sigma = torch.sqrt((1 - alphas_bar_prev) / (1 - alphas_bar)) * torch.sqrt(1 - alphas_bar / alphas_bar_prev)
        
        
    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                extract(self.coeff1, t, x_t.shape) * x_t -
                extract(self.coeff2, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_mean_variance(self, t, x_t, condition, targets, coords, input_lenths, tif):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps, target_list_full, coord_list_full, mask = self.model(x_t, targets, coords, input_lenths, condition, t, tif)
        # eps = torch.randn_like(eps)
        # zeros = torch.zeros(eps.shape[0], eps.shape[1], 2)  # 创建一个全为 0 的张量
        # expanded_eps = torch.cat([eps, zeros], dim=2)
        # eps[:, :, 0:2] = 0
        # print('eps, ', eps.shape)
        
        return eps, var, mask

    def forward(self, x_T, condition, targets, coords, input_lenths, tif):
        """
        Algorithm 2.
        """
        device = x_T.device
        # x_t = torch.randn_like(x_T).to(device)
        x_t = x_T 
        # x_t = x_T[:, :, 0:1]
        loop = 0
        _start = time.time()
        model_time = 0.
        param_time = 0.
        batch_size = x_T.shape[0]
        ts = torch.linspace(self.T, 0, (self.ddim_step + 1)).to(device).to(torch.long)
        for i in range(1, self.ddim_step + 1):
            # print('time step: ', time_step)
            start = time.time()
            cur_t = ts[i - 1] - 1
            prev_t = ts[i] - 1
            t_tensor = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * cur_t
            prev_t_tensor = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * prev_t

            eps, var, mask = self.p_mean_variance(x_t=x_t, t=t_tensor, condition=condition, targets=targets, coords=coords,
                                            input_lenths=input_lenths, tif=tif)
            param_time = time.time()
            # no noise when t == 0
            if cur_t > 0:
                noise = x_t
            else:
                noise = 0
            alpha_bar = extract(self.alphas_cumprod, t_tensor, x_t.shape)
            alpha_bar_prev = extract(self.alphas_cumprod, prev_t_tensor, x_t.shape) if prev_t >= 0 else 1
            sigma = (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev) * self.eta
            # sigma = 0
            first_term = (alpha_bar_prev / alpha_bar) ** 0.5 * x_t
            second_term = ((1 - alpha_bar_prev - sigma) ** 0.5 - (
                    alpha_bar_prev * (1 - alpha_bar) / alpha_bar) ** 0.5) * eps
            third_term = sigma ** 0.5 * noise
            x_t = first_term + second_term + third_term
            # x_t[:, :, 0:2] = x_T[:, :, 0:2]
            model_time += time.time() - start
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."

        x_0 = x_T
        print('sample finished, time elapse:', time.time() - _start)

        return x_0, mask 


