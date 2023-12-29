import torch
import os
import random
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from osgeo import gdal
import concurrent.futures
import torch.nn.functional as F


def encode_and_bind(original_dataframe, feature_to_encode, possible_values):
    for value in possible_values:
        original_dataframe.loc[:, feature_to_encode + '_' + str(value)] = (
                    original_dataframe[feature_to_encode] == value).astype(int)
    res = original_dataframe.drop([feature_to_encode], axis=1)
    return res


class IntpDataset(Dataset):
    def __init__(self, settings, mask_distance, call_name):
        # save init parameters
        self.origin_path = settings['origin_path']
        self.time_fold = settings['fold']
        self.holdout = settings['holdout']
        self.mask_distance = mask_distance
        self.lowest_rank = settings['lowest_rank']
        self.call_name = call_name
        self.debug = settings['debug']

        # load norm info
        with open(self.origin_path + f'Folds_Info/norm_{self.time_fold}.info', 'rb') as f:
            self.dic_op_minmax, self.dic_op_meanstd = pickle.load(f)

        # Generate call_scene_list from 'divide_set_{time_fold}.info', then generate call_list from call_scene_list
        with open(self.origin_path + f'Folds_Info/divide_set_{self.time_fold}.info', 'rb') as f:
            divide_set = pickle.load(f)
        if call_name == 'train':
            # for training set, call_list is same as call_scene_list, because call item will be chosen randomly
            call_scene_list = divide_set[0]
            if self.debug and len(call_scene_list) > 2000:
                call_scene_list = call_scene_list[:2000]
            # do normalization in the parallel fashion
            self.total_df_dict = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_child, file_name) for file_name in call_scene_list]
                for future in concurrent.futures.as_completed(futures):
                    file_name, file_content = future.result()
                    self.total_df_dict[file_name] = file_content
            self.call_list = []
            for scene in call_scene_list:
                df = self.total_df_dict[scene]
                all_candidate = df[df['op'] == 'mcpm10']
                for index in all_candidate.index.tolist():
                    for mask_buffer in range(51):
                        self.call_list.append([scene, index, mask_buffer])
        elif call_name == 'test':
            # for Early Stopping set, call_list is 3 label stations excluding holdout
            call_scene_list = divide_set[1]
            if self.debug and len(call_scene_list) > 2000:
                call_scene_list = call_scene_list[:2000]
            # do normalization in the parallel fashion
            self.total_df_dict = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_child, file_name) for file_name in call_scene_list]
                for future in concurrent.futures.as_completed(futures):
                    file_name, file_content = future.result()
                    self.total_df_dict[file_name] = file_content
            if 'Dataset_res250' in self.origin_path:
                stations = [0, 1, 2, 3]
                stations.remove(self.holdout)
            elif 'LUFT_res250' in self.origin_path:
                label_stations = [0, 1, 2, 3, 4, 5]
                stations = [x for x in label_stations if x not in self.holdout]
            self.call_list = []
            for scene in call_scene_list:
                for station in stations:
                    self.call_list.append([scene, station])
        elif call_name == 'eval':
            # for Final Evaluation set, call_list is holdout station
            call_scene_list = divide_set[1]
            if self.debug and len(call_scene_list) > 2000:
                call_scene_list = call_scene_list[:2000]
            else:
                if 'Dataset_res250' in self.origin_path:
                    call_scene_list = call_scene_list[:4096]
                elif 'LUFT_res250' in self.origin_path:
                    call_scene_list = call_scene_list[:15360]
            # do normalization in the parallel fashion
            self.total_df_dict = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_child, file_name) for file_name in call_scene_list]
                for future in concurrent.futures.as_completed(futures):
                    file_name, file_content = future.result()
                    self.total_df_dict[file_name] = file_content
            self.call_list = []
            if 'Dataset_res250' in self.origin_path:
                for scene in call_scene_list:
                    self.call_list.append([scene, self.holdout])
            elif 'LUFT_res250' in self.origin_path:
                for scene in call_scene_list:
                    for station in self.holdout:
                        self.call_list.append([scene, station])

        print(len(self.total_df_dict.keys()))

        # process the land-use tif
        geo_file = gdal.Open(self.origin_path + 'CWSL_norm.tif')
        tif_channel_list = []
        for i in range(geo_file.RasterCount):
            tif_channel_list.append(np.array(geo_file.GetRasterBand(i + 1).ReadAsArray(), dtype="float32"))
        self.tif = torch.from_numpy(np.stack(tif_channel_list, axis=0))
        self.width = geo_file.RasterXSize
        self.height = geo_file.RasterYSize

        # list interested ops
        self.op_dic = {'mcpm10': 0, 'mcpm2p5': 1, 'ta': 2, 'hur': 3, 'plev': 4, 'precip': 5, 'wsx': 6, 'wsy': 7,
                       'globalrad': 8, }
        self.possible_values = list(self.op_dic.keys())
        self.possible_values.sort()

    def __len__(self):
        return len(self.call_list)

    def process_child(self, filename):
        df = pd.read_csv(self.origin_path + 'Dataset_Separation/' + filename, sep=';')
        # drop everything in bad quality
        df = df[df['Thing'] >= self.lowest_rank]
        # normalize all values (coordinates will be normalized later)
        df = self.norm(d=df)

        return filename, df

    def norm(self, d):
        d_list = []
        for op in d['op'].unique():
            d_op = d.loc[d['op'] == op].copy(deep=True)
            if op in ['s_label_0', 's_label_1', 's_label_2', 's_label_3', 's_label_4', 's_label_5', 's_label_6',
                      'p_label']:
                op_norm = 'mcpm10'
            else:
                op_norm = op
            if op_norm in self.dic_op_minmax.keys():
                d_op['Result_norm'] = (d_op['Result'] - self.dic_op_minmax[op_norm][0]) / (
                            self.dic_op_minmax[op_norm][1] - self.dic_op_minmax[op_norm][0])
            elif op_norm in self.dic_op_meanstd.keys():
                d_op['Result_norm'] = (d_op['Result'] - self.dic_op_meanstd[op_norm][0]) / self.dic_op_meanstd[op_norm][
                    1]
            d_list.append(d_op)
        return pd.concat(d_list, axis=0, ignore_index=False)

    def distance_matrix(self, x0, y0, x1, y1):
        obs = np.vstack((x0, y0)).T
        interp = np.vstack((x1, y1)).T
        d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
        d1 = np.subtract.outer(obs[:, 1], interp[:, 1])
        # calculate hypotenuse
        return np.hypot(d0, d1)

    def idw_interpolation(self, x, y, values, xi, yi, p=2):
        dist = self.distance_matrix(x, y, xi, yi)
        # In IDW, weights are 1 / distance
        weights = 1.0 / (dist + 1e-12) ** p
        # Make weights sum to one
        weights /= weights.sum(axis=0)
        # Multiply the weights for each interpolated point by all observed Z-values
        return np.dot(weights.T, values)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get call_item and scenario_story
        if self.call_name == 'train':
            # for training set, call item is randomly taken from the 'mcpm10'
            #     - get the scenario, filter with lowest reliable sensor rank
            scene = self.call_list[idx][0]
            random_index = self.call_list[idx][1]

            df = self.total_df_dict[scene]
            #     - get a random call_item, those remianed are story
            all_candidate = df[df['op'] == 'mcpm10']

            random_row = all_candidate.loc[random_index]
            call_item = pd.DataFrame([random_row], index=[random_index])
            remaining_candidate = all_candidate.drop(random_index)
            rest_story = df.loc[(df['op'].isin(self.op_dic.keys())) & (df['op'] != 'mcpm10')]
            df_story = pd.concat([remaining_candidate, rest_story], axis=0, ignore_index=True)
        elif self.call_name in ['test', 'eval']:
            # for Early Stopping set and Final Evaluation set
            #     - get the scenario and target station, filter with lowest reliable sensor rank
            scene = self.call_list[idx][0]
            target = self.call_list[idx][1]
            df = self.total_df_dict[scene]
            #     - get call_item by target station, story are filtered with op_dic
            call_item = df[df['op'] == f's_label_{target}']
            if len(call_item) != 1:
                call_item = call_item[0]
            df_story = df.loc[df['op'].isin(self.op_dic.keys())]

        # processing senario informations:
        #     - mask out all readings within 'mask_distance'
        if self.mask_distance == -1:
            this_mask = self.call_list[idx][2]
        else:
            this_mask = self.mask_distance
        df_filtered = df_story.loc[(abs(df_story['Longitude'] - call_item.iloc[0, 2]) + abs(
            df_story['Latitude'] - call_item.iloc[0, 3])) >= this_mask, :].copy()
        df_filtered = df_filtered.drop(columns=['Result'])

        #     - generate normalized story token serie [value, rank, loc_x, loc_y, one_hot_op]
        df_filtered = df_filtered[['Result_norm', 'Thing', 'Longitude', 'Latitude', 'op']]

        # hour = float(scene.split('T')[1].split('-')[0]) / 24.0
        # new_row = [hour, 3, 125, 125, 'globalrad']
        # df_filtered = pd.concat([df_filtered, pd.DataFrame([new_row], columns=df_filtered.columns)], ignore_index=True)

        # value_counts = df_filtered['op'].value_counts()
        # max_count = value_counts.max()
        # repeated_rows = []
        # for value, count in value_counts.items():
        #     repeated_rows.append(df_filtered[df_filtered['op'] == value])
        #     if count < max_count:
        #         rows_to_add = max_count - count
        #         repeated_rows.append(df_filtered[df_filtered['op'] == value].sample(n=rows_to_add, replace=True))
        # df_filtered = pd.concat(repeated_rows, ignore_index=True)

        if self.call_name in ['train', 'train_self']:
            q_candidate = call_item.copy()
            # q_candidate.loc[:, 'Thing'] = np.random.randint(1, 4, size=len(q_candidate))
        else:
            q_candidate = call_item.copy()

        q_loc = torch.from_numpy(q_candidate[['Thing', 'Longitude', 'Latitude']].values).float()
        intp_candidate = df_filtered[df_filtered['op'] == 'mcpm10'].groupby(['Longitude', 'Latitude']).mean(
            numeric_only=True).reset_index()
        interpolated_gird = torch.zeros((len(q_candidate), 1))
        x0, y0 = np.meshgrid(range(0, 250), range(0, 250))
        x0 = x0.flatten()
        y0 = y0.flatten()
        xi = q_candidate['Longitude'].values
        yi = q_candidate['Latitude'].values
        x = intp_candidate['Longitude'].values
        y = intp_candidate['Latitude'].values
        # print(xi,yi)
        values = intp_candidate['Result_norm'].values
        # print(x,y,values)
        interpolated_values = self.idw_interpolation(x, y, values, xi, yi)
        interpolated_grid_values = self.idw_interpolation(x, y, values, x0, y0)
        interpolated_grid = torch.from_numpy(interpolated_values).reshape((len(q_candidate), 1)).float()
        # print(interpolated_grid.shape)
        interpolated_grid_output = torch.from_numpy(interpolated_grid_values).reshape((250, 250)).float()
        op_onehot = torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0, 0]]).float()
        q_serie = torch.cat([interpolated_grid, q_loc, op_onehot], dim=1)
        q_serie[:, 1:4] = torch.div(q_serie[:, 1:4], torch.tensor([3, self.width, self.height]))

        input_serie = torch.from_numpy(encode_and_bind(df_filtered, 'op', self.possible_values).values).float()
        input_serie[:, 1:4] = torch.div(input_serie[:, 1:4], torch.tensor([3, self.width, self.height]))

        # generate answer for the item
        answer = torch.from_numpy(np.array([call_item.loc[:, 'Result_norm'], ])).float()
        answer_x, answer_y = torch.from_numpy(np.array([call_item.loc[:, 'Longitude'], ])).float(), torch.from_numpy(np.array([call_item.loc[:, 'Latitude'], ])).float()
        # print(answer)
        # print(torch.from_numpy(np.array([call_item.loc[:, 'Longitude'], ])).float(), torch.from_numpy(np.array([call_item.loc[:, 'Latitude'], ])).float(), answer)
        return q_serie, input_serie, answer,  self.tif, interpolated_grid_output, (answer_x, answer_y)


def collate_fn(examples):
    q_series = torch.stack([ex[0] for ex in examples], 0)

    input_lenths = torch.tensor([len(ex[1]) for ex in examples])
    input_series = pad_sequence([ex[1] for ex in examples], batch_first=True, padding_value=0.0)

    answers = torch.tensor([ex[2] for ex in examples])

    return q_series, input_lenths, input_series, answers


def collate_fn_img(examples):
    q_series = torch.stack([ex[0] for ex in examples], 0)

    input_lenths = torch.tensor([len(ex[1]) for ex in examples])
    answers = torch.tensor([ex[2] for ex in examples])
    answer_loc = torch.tensor([ex[5] for ex in examples])
    # 使用循环将每个元素中的第二个张量提取并添加到 batch_tensor 中
    input_img = torch.zeros(len(examples), 256, 256)
    # for i in range(len(examples)):
    #     second_tensor = F.pad(examples[i][4], (0, 6, 0, 6))  # 提取第二个元素（第二个张量）
    #     input_img[i, :, :] = second_tensor
    #     # origin_img[i, :, :] = examples[i][6]
    #     # print(second_tensor)
    second_tensors = [example[4] for example in examples]
    stacked_tensors = torch.stack(second_tensors, dim=0)
    padded_tensors = F.pad(stacked_tensors, (0, 6, 0, 6))
    input_img = padded_tensors
    tif = pad_sequence([ex[3] for ex in examples if len(ex[3]) > 2], batch_first=True, padding_value=0.0)
    input_series = pad_sequence([ex[1] for ex in examples], batch_first=True, padding_value=0.0)
    return q_series, input_lenths, input_img.float(), answers, input_series, answer_loc, tif


if __name__ == '__main__':
    # random_tensor = torch.rand(25, 25)
    # random_tensor[0:9, 0:9] = 0
    # random_tensor[20:24, 20:24] = 0
    # # random_tensor = torch.tensor(random_tensor)
    # zero, lst = get_idw_list(random_tensor)
    # output = interpolation(zero, lst)

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
    settings = {
        'origin_path': '../Datasets/Dataset_res250/',
        'debug': False,
        'bp': False,

        'batch': 2,
        'accumulation_steps': 256 // 128,
        'epoch': 200000,
        'trans_lr': 1,  # 1e-7
        'nn_lr': 1,  # 1e-6
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
        "channel": 32,  # tdim = ch*4
        "cdim": 256,  # cdim = vi_dim
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
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
    dataset_train = IntpDataset(settings=settings, mask_distance=-1, call_name='train')
    dataloader_tr = torch.utils.data.DataLoader(dataset_train, batch_size=settings['batch'], shuffle=True,
                                                collate_fn=collate_fn_img, num_workers=10, prefetch_factor=32,
                                                drop_last=True)
    print('data loaded')

    # model_unet = UNet(T=diffusion_configs["T"], ch=diffusion_configs["channel"],
    #                   ch_mult=diffusion_configs["channel_mult"], cdim=diffusion_configs["cdim"],
    #                   attn=diffusion_configs["attn"],
    #                   num_res_blocks=diffusion_configs["num_res_blocks"], dropout=diffusion_configs["dropout"]).to(
    #     device)
    # # model_unet.load_state_dict(torch.load("./Checkpoints/ckpt_49_.pt"))
    # trainer = GaussianDiffusionTrainer(
    #     model_unet, diffusion_configs["beta_1"], diffusion_configs["beta_T"], diffusion_configs["T"]).to(device)
    # # print(trainer)
    # sampler = GaussianDiffusionSampler(
    #     model_unet, diffusion_configs["beta_1"], diffusion_configs["beta_T"], diffusion_configs["T"],
    #     ddim_step=diffusion_configs["ddim_step"]).to(device)
    # print('model loaded')

    # optimizer = optim.AdamW([
    #     {'params': model_unet.time_embedding.parameters(), 'lr': settings['nn_lr'], 'weight_decay': 1e-3},  # 1e-3
    #     {'params': model_unet.upblocks.parameters(), 'lr': settings['trans_lr'], 'weight_decay': 1e-6},  #
    #     {'params': model_unet.downblocks.parameters(), 'lr': settings['nn_lr'], 'weight_decay': 1e-6},  #
    #     {'params': model_unet.middleblocks.parameters(), 'lr': settings['nn_lr'], 'weight_decay': 1e-6},  #
    #     {'params': model_unet.tail.parameters(), 'lr': settings['nn_lr'], 'weight_decay': 1e-3},  #
    #     # {'params': vaemodel.parameters(), }
    # ],
    #     betas=(0.9, 0.95), eps=1e-8
    # )

    # data_iter = iter(dataloader_tr)
    # with torch.no_grad():
    #     for q_tokens_tr, input_lenths_tr, input_series_tr, answers_tr, input_tif in dataloader_tr:
    #         input_series_tr = input_series_tr.float()
    #         input_image = input_series_tr.reshape(settings['batch'], 1, 256, 256).to(device)
    #         # output_ex = sampler(input_image).reshape(settings['batch'], 256, 256).to(device)
    #         break
    # model = Transformer(settings, device)
    i = 0
    for q_tokens_ex, input_lenths_ex, input_img_ex, answers_ex, input_series, answer_loc in dataloader_tr:
        # plt.figure()  # 创建一个新的图像
        # plt.imshow(input_series_ex[0].cpu(), cmap='hot', interpolation='nearest')
        # plt.colorbar()  # 添加颜色条
        # plt.title(f'Heatmap for output Image')  # 设置图像标题
        # plt.show()  # 显示图像
        #
        # plt.figure()  # 创建一个新的图像
        # plt.imshow(origin_img[0].cpu(), cmap='hot', interpolation='nearest')
        # plt.colorbar()  # 添加颜色条
        # plt.title(f'Heatmap for output Image')  # 设置图像标题
        # plt.show()  # 显示图像
        # i += 1
        # if i > 10:
            break
        # print(input_series_ex)

        # input_series_ex = idw_interpolation(input_series_ex)
        # input_series_ex = input_series_ex.float()
        # input_image = input_series_ex.reshape(settings['batch'], 1, 256, 256).to(device)
        # print(i)

        # output_ex = sampler(input_image).reshape(settings['batch'], 256, 256).to(device)
    # while True:
    #     # train 1 iteration
    #     model.train()
    #     with autocast():
    #         real_iter = iter_counter // settings['accumulation_steps'] + 1
    #
    #         try:
    #             batch = next(data_iter)
    #         except StopIteration:
    #             data_iter = iter(dataloader_tr)
    #             batch = next(data_iter)
    #         q_tokens_tr, input_lenths_tr, input_series_tr, answers_tr, _ = batch
    #         output_tr, mu_tr, var_tr = model(q_tokens_tr.to(device), input_lenths_tr.to(device),
    #                                          input_series_tr.to(device))