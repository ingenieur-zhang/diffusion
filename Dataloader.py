import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from osgeo import gdal
import vae as v
import Transformer as tr
import os, glob, inspect, time, math, torch, pickle
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return res


class Tag:
    def __init__(self, id):
        self.id = id

    def __getitem__(self, item):
        print('这个方法被调用')
        return self.id

class IntpDataset(Dataset):
    def __init__(self, origin_path, call_path, call_name, debug):
        self.origin_path = origin_path
        self.call_path = call_path
        self.call_name = call_name
        self.debug = debug
        
        self.call_list = pd.read_csv(self.call_path + self.call_name + '.csv', sep=';')
        if debug and len(self.call_list) > 2000:
            self.call_list = self.call_list[:2000]
        
        geo_file = gdal.Open(self.origin_path + 'CWSL_norm.tif')
        tif_channel_list = []
        for i in range(geo_file.RasterCount):
            tif_channel_list.append(np.array(geo_file.GetRasterBand(i + 1).ReadAsArray(), dtype="float32"))
        self.tif = torch.from_numpy(np.stack(tif_channel_list, axis=0))
        self.width = geo_file.RasterXSize
        self.height = geo_file.RasterYSize

        self.op_dic = {'mcpm10': 0, 'mcpm2p5': 1, 'ta': 2, 'hur': 3, 'plev': 4, 'precip': 5, 
                       'wdir': 6, 'wspeed': 7, 'globalrad': 8, }

    def __len__(self):
        return len(self.call_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        call_item = self.call_list.iloc[idx:idx+1]
        df = pd.read_csv(self.origin_path + call_item.iloc[0, 4] + '.csv', sep=';')
        df_story = df.loc[df['op'].isin(self.op_dic.keys())]
        df_label = call_item[['op', 'Longitude', 'Latitude', 'Result']]
        df_label.iloc[0, 3] = 0.0
        df_label.iloc[0, 0] = 'mcpm10'
        df_input = pd.concat([df_label, df_story], axis=0)
        df_input = encode_and_bind(df_input, 'op')
        input_serie = torch.from_numpy(df_input.values).float()
        input_serie[:, 0] = input_serie[:, 0] / self.width
        input_serie[:, 1] = input_serie[:, 1] / self.height
        answer = torch.from_numpy(np.array([call_item.iloc[0, 3], ])).float()
        # vae =  v.VAE()
        # answer, mu, log  = vae(answer)
        return input_serie, answer
    

def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [ex[0] for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples])
    # 对batch内的样本进行padding，使其具有相同长度
    inputs = pad_sequence(inputs, batch_first=True, padding_value=-200.0)
    return inputs, lengths, targets


#########################################################################################
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
dataloader_tr = torch.utils.data.DataLoader(dataset_train, batch_size=settings['batch'], shuffle=True, collate_fn=collate_fn, num_workers=4, prefetch_factor=4)
model = tr.Transformer(
        word_col=settings['word_col'], pos_col=settings['pos_col'], embedding_dim=settings['embedding_dim'],
        feedforward_dim=settings['feedforward_dim'], output_dim=settings['output_dim'],
        num_head=settings['num_head'], num_layers=settings['num_layers'],
        dropout=0.1, activation="gelu", device=device
    )
# for inputs_ex, lengths_ex, targets_ex in dataloader_tr:
for inputs_ex, lengths_ex, targets_ex in dataloader_tr:
    output_vae = model_vae(inputs_ex.to(device))

    output_ex = model(inputs_ex.to(device), lengths_ex.to(device))

print(dataset_train[1])
