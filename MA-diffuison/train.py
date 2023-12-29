import os, time
import torch
import argparse
import itertools
import numpy as np
from Trans_scrpit import offlinesolver as solver
from tqdm import tqdm
import torch.optim as optim
from Diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size
from Datasets import Dataloader as dl
import pickle
from Diffusion.Model import UNet


settings = {
    'origin_path': './Datasets/Original_res250/',
    'call_path': './Datasets/Trans_Call_res250/',
    'debug': False,
    'bp': False,

    'batch': 4,
    'epoch': 1000,
    'lr': 1e-5,


    'word_col': [0, 1, 2],
    'pos_col': [3, 4, 5, 6, 7, 8, 9, 10, 11],
    'embedding_dim': 512,
    'feedforward_dim': 2048,
    'output_dim': 1,
    'num_head': 8,
    'num_layers': 6,
    'classifier_scale': 0.1,
    'timestep': None,
}
timestep = torch.arange(100)
settings['timestep'] = timestep
def make_dir(path):
    try:
        os.mkdir(path)
    except:
        pass

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
with open(settings['origin_path'] + 'norm.info', 'rb') as f:
    dic_op_minmax = pickle.load(f)
# model_unet = UNetModel(
#     in_channels = 1,
#     model_channels = 256,  # 256*12
#     out_channels = 1,
#     num_res_blocks=1,
#     attention_resolutions=tuple("32,16,8"),
#     dropout=0.1,
#     num_classes=None,
#     channel_mult= (1, 2, 4, 8),  # channel_mult=(1, 1, 2, 2, 4, 4),  # 根据model creation里的定义，input size=256对应的大小
#     conv_resample=True,
#     dims=2,
#     use_checkpoint=False,
#     use_fp16=False,
#     num_heads=1,
#     num_head_channels=-1,
#     num_heads_upsample=-1,
#     use_scale_shift_norm=False,
#     resblock_updown=False,
#     encoder_channels=None,
# ).to(device)
modelConfig = {
        "state": "train", # or eval
        "epoch": 200,
        "batch_size": settings['batch'],
        "T": 1000,
        "channel": 32,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8
        }
initial_usage = torch.cuda.memory_allocated()
# print("0", initial_usage)  # 0
model_unet = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
# print(model_unet)
# model_diffusion = GaussianDiffusion(
#     noise_schedule = "squaredcos_cap_v2", steps = 100, model=model_unet, device=device
# )
# print(model_diffusion)
trainer = GaussianDiffusionTrainer(
        model_unet, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
# print(trainer)
sampler = GaussianDiffusionSampler(
            model_unet, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
# print(sampler)
dataset_train = dl.IntpDataset(origin_path=settings['origin_path'], call_path=settings['call_path'], call_name='train',
                               debug=settings['debug'])
dataloader_tr = torch.utils.data.DataLoader(dataset_train, batch_size=settings['batch'], shuffle=True,
                                            collate_fn=dl.collate_fn, num_workers=4, prefetch_factor=4)

dataset_test = dl.IntpDataset(origin_path=settings['origin_path'], call_path=settings['call_path'], call_name='test',
                              debug=settings['debug'])
dataloader_ex = torch.utils.data.DataLoader(dataset_test, batch_size=settings['batch'], shuffle=False,
                                            collate_fn=dl.collate_fn, num_workers=4, prefetch_factor=4)

dataloader_tr_pm = torch.utils.data.DataLoader(dataset_train, batch_size=settings['batch'], shuffle=True,
                                               collate_fn=dl.collate_fn_img, num_workers=4, prefetch_factor=4)

# for inputs_ex, lengths_ex, targets_ex in dataloader_ex:
optimizer = torch.optim.AdamW(
    itertools.chain(
        model_unet.parameters()
    ),
    lr=settings['lr'],
    weight_decay=1e-4
)

epochs = settings['epoch']
batch_size = settings['batch']

print("\nTraining to %d epochs (%d of minibatch size)" % (epochs, batch_size))

# x = torch.randn(batch_size, 3, 128, 128).to(device)
# t = torch.randint(1000, (batch_size, )).to(device)
# y = model_unet(x, t)
# print(y.shape)
model_init_usage = torch.cuda.memory_allocated()
# print("0", model_init_usage)  # 0
for epoch in range(epochs):
    # '''

    for inputs_pm, lengths_pm, targets_pm in dataloader_tr_pm:
        inputs_pm = inputs_pm.float()
        input_image = inputs_pm.reshape(settings['batch'], 1, 8, 8).to(device)
        # print(input_image)
        dataloader_usage = torch.cuda.memory_allocated()
        # print("0", dataloader_usage)  # 0
        output_pm = sampler(input_image)
        # print(output_pm)

        # print(targets_pm)
        # test_means_origin = output_pm * dic_op_minmax['mcpm10'][1] + dic_op_minmax['mcpm10'][0]
        # test_y_origin = targets_pm * dic_op_minmax['mcpm10'][1] + dic_op_minmax['mcpm10'][0]
    # '''

    total_loss = 0
    mini_batch = 0
    for inputs_pm, lengths_pm, targets_pm in dataloader_tr_pm:
        input_image = inputs_pm.reshape(settings['batch'], 1, 8, 8).to(device)
        # print(outputpm)
        # '''
        batch_loss = trainer(input_image).sum() / 1000.
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_unet.parameters(), 0.02)
        if mini_batch % 50 == 0:
             solver.plot_grad_flow(model_unet.named_parameters())
        optimizer.step()
        print(f'\tEpoch {epoch}, Iter {mini_batch} - Loss: {batch_loss.item()}')
        mini_batch += 1
        total_loss += batch_loss.item()
        # output_ex_t = model_unet(input_image, t_in_unet)
        # input_image = input_image - output_ex_t
        # output_ex.append(output_ex_t)
        print(f"Epoch {epoch} - Loss: {total_loss:.3f}")
        torch.save(model_unet.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(epoch) + "_.pt"))
        # '''

make_dir(path="results")
result_list = ["diffusion_unet", ]
for result_name in result_list:
    make_dir(path=os.path.join("results", result_name))






