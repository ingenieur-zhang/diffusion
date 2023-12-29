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
    # print(v.device, t.device)
    # t = t.to(device)
    device = t.device
    v = v.to(device)
    out = torch.gather(v, index=t, dim=0).to(device).float()
    # out = torch.gather(v, index=t, dim=0).to(device).float()  # unextended
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

    def forward(self, x_0, condition, tif):
        """
        Algorithm 1.
        """
        time_batch  = 8
        device=x_0.device
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        # extended_t = torch.randint(self.T, size=(x_0.shape[0]*time_batch, ), device=x_0.device)
        # extended_t = torch.round(torch.clamp(torch.abs(torch.normal(mean=0, std=300, size=(x_0.shape[0] * time_batch,), device=x_0.device)), max=999)).long() # 修改t的采样分布
        noise = torch.randn_like(x_0)

        # extended_noise = noise.unsqueeze(0).expand(time_batch, *x_0.shape)
        # extended_x0 = x_0.unsqueeze(0).expand(time_batch, *x_0.shape)
        # extended_t = torch.randint(self.T, size=(x_0.shape[0], time_batch), device=x_0.device)
        # print(extended_x0.shape)
        # extended_sqrt_alphas_bar = self.sqrt_alphas_bar.unsqueeze(0).expand(time_batch, *self.sqrt_alphas_bar.shape)
        # extended_sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar.unsqueeze(0).expand(time_batch, *self.sqrt_one_minus_alphas_bar.shape)
        # print(extended_sqrt_one_minus_alphas_bar.shape)

        coeff1 = extract(self.sqrt_alphas_bar, t, x_0.shape)
        coeff2 = extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape)

        # extended_coeff1 = coeff1.unsqueeze(0).expand(time_batch, *coeff1.shape)
        # extended_coeff2 = coeff2.unsqueeze(0).expand(time_batch, *coeff2.shape)

        # coeff1 = extract(self.sqrt_alphas_bar, extended_t, x_0.shape)
        # print(coeff1.shape)
        # extended_coeff1 = coeff1.reshape(time_batch, x_0.shape[0], 1,1,1)
        # print(coeff1.shape))
        # coeff2 = extract(self.sqrt_one_minus_alphas_bar, extended_t, x_0.shape)
        # shape2 = coeff2.shape
        # extended_coeff2 = coeff2.reshape(time_batch, x_0.shape[0], 1,1,1)
        # print(extended_sqrt_alphas_bar.shape)

        x_t = (
                extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        # extended_xt = (
                # extended_coeff1 * extended_x0 +
                # extended_coeff2 * extended_noise)

        # loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')  # 查表求出xt，
        # extended_xt = extended_xt.reshape((time_batch*x_0.shape[0], 1, extended_xt.shape[3], extended_xt.shape[4]))
        # extended_t = extended_t.reshape((time_batch*t.shape[0], t.shape[1]))

        # extended_condition = condition.expand(time_batch, *condition.shape).reshape((time_batch*condition.shape[0], condition.shape[1]))
        # extended_noise = extended_noise.reshape((time_batch*extended_noise.shape[1], 1, extended_noise.shape[3], extended_noise.shape[4]))
        # print(extended_xt.shape)
        # print(extended_t.shape)
        # print(extended_condition.shape)
        # print(extended_noise.shape)a
        loss = F.mse_loss(self.model(x_t.to(device), t.to(device), condition.to(device), tif.to(device)), noise.to(device), reduction='none')  # 查表求出xt，
        # loss = 0.5 * F.mse_loss(self.model(x_t, t, condition), noise, reduction='none') + 0.5 * F.mse_loss(
        #     self.model(x_t, t, condition), x0, reduction='none')  # 查表求出xt，
        # loss = loss.view(time_batch, *x_0.shape).sum(dim=0)
        return loss 


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, ddim_step, eta):
        super().__init__()

        self.model = model
        self.T = T
        self.ddim_step = ddim_step
        self.eta = eta
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
        # device = x_t.device
        # t = t.to(device)
        # pred_xstart = pred_xstart.to(device)
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_mean_variance(self, x_t, t, condition, tif):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        # print(x_t.shape)
        eps = self.model(x_t, t, condition, tif)
        # print(eps.shape)
        # xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return eps, var

    def forward(self, x_T, condition, tif):
        """
        Algorithm 2.
        """
        #         current_directory = os.getcwd()
        #         print(current_directory)
        #         file_path = './1.jpg'  # 替换为你的 PNG 文件路径

        # # 使用 torchvision 的 transforms 进行图像预处理
        #         preprocess = transforms.Compose([transforms.ToTensor(),])  # 将图像转换为 PyTorch 张量

        # # 读取 PNG 文件并进行预处理
        #         image = Image.open(file_path)
        #         image_tensor = preprocess(image)
        #         extracted_tensor = image_tensor[2]

        # # 复制提取的部分，拼成一个 256x256 的张量
        #         expanded_tensor = extracted_tensor.repeat(4, 4)

        # # 复制四份并改变形状成 4x1x256x256 的张量
        #         final_tensor = expanded_tensor.unsqueeze(0).repeat(4, 1, 1, 1).to(x_T.device)
        #         print(final_tensor.shape)
        #         x_t = final_tensor
        device = x_T.device
        # x_t = torch.randn(x_T.shape[0], 1, 256, 256).to(device)
        x_t = x_T
        # plt.figure()  # 创建一个新的图像
        # plt.imshow(x_T[0][0].cpu(), cmap='hot', interpolation='nearest')
        # plt.colorbar()  # 添加颜色条
        # plt.title(f'Heatmap for input Image')  # 设置图像标题
        # plt.savefig(f' original_image_output.png')
        # plt.show()  # 显示图像
        loop = 0
        _start = time.time()
        model_time = 0.
        param_time = 0.
        batch_size = x_T.shape[0]
        ts = torch.linspace(self.T, 0, (self.ddim_step + 1)).to(device).to(torch.long)
        # print(x_T)
        # for time_step in reversed(range(self.ddim_step)):
        for i in range(1, self.ddim_step + 1):
            # print('time step: ', time_step)
            start = time.time()
            cur_t = ts[i - 1] - 1
            prev_t = ts[i] - 1
            # print(cur_t, prev_t)
            # t_tensor = torch.tensor([cur_t] * batch_size,
            #                     dtype=torch.long).to(device).unsqueeze(1)

            t_tensor = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * cur_t
            prev_t_tensor = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * prev_t
            # t = t.to(device)
            eps, var = self.p_mean_variance(x_t=x_t, t=t_tensor, condition=condition, tif=tif)
            # print(x_t[0])
            # print('model time:', time.time()-start)
            param_time = time.time()
            # no noise when t == 0
            if cur_t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0

            # print(x_t.device, t.device, mean.device)
            # eps = self._predict_eps_from_xstart(x_t, t, mean)
            alpha_bar = extract(self.alphas_cumprod, t_tensor, x_t.shape)
            alpha_bar_prev = extract(self.alphas_cumprod, prev_t_tensor, x_t.shape) if prev_t >= 0 else 1
            # sigma = (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev) * self.eta
            sigma = 0
            first_term = (alpha_bar_prev / alpha_bar) ** 0.5 * x_t
            second_term = ((1 - alpha_bar_prev - sigma) ** 0.5 - (
                        alpha_bar_prev * (1 - alpha_bar) / alpha_bar) ** 0.5) * eps
            # print((1 - alpha_bar_prev - sigma)**0.5)
            third_term = sigma ** 0.5 * noise
            # print((alpha_bar_prev / alpha_bar)**0.5)
            # print(((1 - alpha_bar_prev - sigma)**0.5 -(alpha_bar_prev * (1 - alpha_bar) / alpha_bar)**0.5))
            # print(sigma**0.5)
            x_t = first_term + second_term + third_term
            # x_n = x_t.squeeze()
            # plt.figure()  # 创建一个新的图像
            # plt.imshow(x_n[0].cpu(), cmap='hot', interpolation='nearest')
            # plt.colorbar()  # 添加颜色条
            # plt.title(f'Heatmap for input Image')  # 设置图像标题
            # plt.savefig(f' heatmap_image_output{i}.png')
            # plt.show()  # 显示图像
            # print(third_term)
            # print('finish')
            model_time += time.time() - start
            # x_t = mean + torch.sqrt(var) * noise
            cal_usage = torch.cuda.memory_allocated()
            # print('calculating usage: ', cal_usage)  #
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."

        x_0 = x_t
        print('sample finished, time elapse:', time.time() - _start)
        # print('model time:', model_time)
        # print('param time:', param_time)
        return x_0  # torch.clip(x_0, -1, 1)

if __name__ == '__main__':
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        ngpu = 0
        print(f'Working on CPU')
    else:
        device = torch.device("cuda")
        ngpu = torch.cuda.device_count()
        if ngpu > 1:
            device_list = [i for i in range(ngpu)]
            print(f'Working on multi-GPU {device_list}')
        else:
            print(f'Working on single-GPU')
    batch_size = 2
    settings = {
        'origin_path': '../Datasets/Dataset_res250/',
        'debug': False,
        'bp': False,
        'debug_stage': 1,

        'batch': 2,
        'accumulation_steps': 256 // 128,
        'epoch': 100,
        'trans_lr': 1e-7,
        'nn_lr': 1e-6,
        'es_mindelta': 1.0,
        'es_endure': 50,

        'word_col': [0, 1],
        'pos_col': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'q_col': 3,

        'embedding_dim': 32,
        'feedforward_dim': 4 * 32,
        'num_head': 2,
        'num_layers': 4,
        'output_hidden_layers': 2,
        'vi_dim': 256,
        'dropout': 0.3,

        'fold': 0,
        'holdout': 0,
        'lowest_rank': 1,
    }
    diffusion_configs = {
        "state": "train",  # or eval
        "epoch": 200,
        "batch_size": settings['batch'],
        "T": 1000,
        "channel": 32,
        "cdim": 32,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-3,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "schedule": "cosine",
        "ddim_step": 20,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",  ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8
    }
    # dataset_train = IntpDataset(settings=settings, mask_distance=-1, call_name='test')
    # dataloader_tr = torch.utils.data.DataLoader(dataset_train, batch_size=settings['batch'], shuffle=True,
    #                                             collate_fn=collate_fn_img, num_workers=10, prefetch_factor=32,
    #                                             drop_last=True)
    # print('data loaded')
    # model = INTP_Transformer(settings, device)
    model = UNet(T=diffusion_configs["T"], ch=diffusion_configs["channel"], ch_mult=diffusion_configs["channel_mult"],
         attn=diffusion_configs["attn"], cdim=diffusion_configs["cdim"],
         num_res_blocks=diffusion_configs["num_res_blocks"], dropout=diffusion_configs["dropout"]).to(device)
    trainer = GaussianDiffusionTrainer(model=model, beta_1=diffusion_configs['beta_1'], beta_T=diffusion_configs['beta_T'], T=1000)

    q = torch.randn(batch_size, 1, 256, 256)
    c = torch.randn(batch_size, 32)

    y = trainer(q,c)
    #     #
    print(y.shape)
