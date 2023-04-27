import math
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def length_to_mask(lengths, total_len, device):
    max_len = total_len
    mask = torch.arange(max_len).expand(lengths.shape[0], max_len).to(device) < lengths.unsqueeze(1)
    return mask


class Transformer(nn.Module):
    def __init__(self, word_col, pos_col, embedding_dim, 
                 feedforward_dim, output_dim, 
                 num_head, num_layers, 
                 dropout, activation, device):
        super(Transformer, self).__init__()
        self.device = device
        # 词嵌入 + 位置嵌入，先尝试同时学习
        self.word_col = word_col
        self.pos_col = pos_col
        self.embedding_dim = embedding_dim
        self.embedding_layer = torch.nn.Sequential(
            torch.nn.Linear(len(self.word_col) + len(self.pos_col), 32),
            torch.nn.LeakyReLU(0.25),
            torch.nn.Linear(32, 128),
            torch.nn.LeakyReLU(0.25),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, embedding_dim),
        ).to(self.device)
        
        # 编码层：使用Transformer
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_head, feedforward_dim, dropout, activation, batch_first=True).to(self.device)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers).to(self.device)

        # 输出层
        self.output = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 1024),
            torch.nn.LeakyReLU(0.25),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 2048),
            torch.nn.LeakyReLU(0.25),
            torch.nn.BatchNorm1d(2048),
            torch.nn.Linear(2048, 512),
            torch.nn.LeakyReLU(0.25),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 128),
            torch.nn.LeakyReLU(0.25),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 16),
            torch.nn.LeakyReLU(0.25),
            torch.nn.BatchNorm1d(16),
            torch.nn.Linear(16, output_dim),
        ).to(self.device)

    def forward(self, inputs, lengths):
        total_length = inputs.size(1)
        print(len(inputs))
        embedded_batch_list = []
        for i in range(len(inputs)):
            embedded_batch_list.append(self.embedding_layer(inputs[i]))
        embedded_batch = torch.stack(embedded_batch_list, 0)
        # embedded_batch = inputs.repeat(1, 1, 8)
        
        attention_mask = length_to_mask(lengths, total_length, self.device) == False
        # 根据批次batch中每个序列长度生成mask矩阵
        hidden_states = self.transformer(embedded_batch, src_key_padding_mask=attention_mask)
        hidden_states = hidden_states[:, 0, :]
        # 取第一个标记的输出结果作为分类层的输入
        output = self.output(hidden_states)

        return output


class Transformer_gdm(nn.Module):
    def __init__(self, word_col, pos_col, embedding_dim,
                 feedforward_dim, output_dim,
                 num_head, num_layers,
                 dropout, activation, device):
        super(Transformer_gdm, self).__init__()
        self.device = device
        # 词嵌入 + 位置嵌入，先尝试同时学习
        self.word_col = word_col
        self.pos_col = pos_col
        self.embedding_dim = embedding_dim
        self.embedding_layer = torch.nn.Sequential(
            torch.nn.Linear(len(self.word_col) + len(self.pos_col), 32),
            torch.nn.LeakyReLU(0.25),
            torch.nn.Linear(32, 128),
            torch.nn.LeakyReLU(0.25),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, embedding_dim),
        ).to(self.device)

        # 编码层：使用Transformer
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_head, feedforward_dim, dropout, activation,
                                                   batch_first=True).to(self.device)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers).to(self.device)

        # 输出层
        self.output = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 1024),
            torch.nn.LeakyReLU(0.25),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 2048),
            torch.nn.LeakyReLU(0.25),
            torch.nn.BatchNorm1d(2048),
            torch.nn.Linear(2048, 512),
            torch.nn.LeakyReLU(0.25),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 128),
            torch.nn.LeakyReLU(0.25),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 16),
            torch.nn.LeakyReLU(0.25),
            torch.nn.BatchNorm1d(16),
            torch.nn.Linear(16, output_dim),
        ).to(self.device)

    def forward(self, inputs, lengths):  # input 256*n*12
        total_length = inputs.size(1)

        embedded_batch_list = []
        for i in range(len(inputs)):
            embedded_batch_list.append(self.embedding_layer(inputs[i]))  # embedding 输入n*12
        embedded_batch = torch.stack(embedded_batch_list, 0)
        # embedded_batch = inputs.repeat(1, 1, 8)

        attention_mask = length_to_mask(lengths, total_length, self.device) == False
        # 根据批次batch中每个序列长度生成mask矩阵
        hidden_states = self.transformer(embedded_batch, src_key_padding_mask=attention_mask)
        hidden_states = hidden_states[:, 0, :]
        # 取第一个标记的输出结果作为分类层的输入
        output = self.output(hidden_states)

        return output
