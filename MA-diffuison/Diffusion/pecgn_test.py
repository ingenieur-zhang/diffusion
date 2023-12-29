import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, knn_graph
from torch_geometric.utils import to_dense_adj
import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn import init
from scipy import sparse
import torch.nn.parallel
import torch.utils.data
from Diffusion.spatial_utils import *
import time
from Diffusion.layers import GraphAttentionLayer, SpGraphAttentionLayer


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
    def forward(self, x):
        # return self.leaky_relu(x)
        return x * torch.sigmoid(x)

def length_to_mask(lengths, total_len, device):
    max_len = total_len
    mask = torch.arange(max_len).expand(lengths.shape[0], max_len).to(device) < lengths.unsqueeze(1)
    mask = mask.int()
    return mask


def padded_seq_to_vectors(padded_seq, logger):
    # Get the actual lengths of each sequence in the batch
    actual_lengths = logger.int()
    # Step 1: Form the first tensor containing all actual elements from the batch
    mask = torch.arange(padded_seq.size(1), device=padded_seq.device) < actual_lengths.view(-1, 1)
    # print(mask)
    tensor1 = torch.masked_select(padded_seq, mask.unsqueeze(-1)).view(-1, padded_seq.size(-1))
    # Step 2: Form the second tensor to record which row each element comes from
    tensor2 = torch.repeat_interleave(torch.arange(padded_seq.size(0), device=padded_seq.device), actual_lengths)
    return tensor1, tensor2, mask  # 真实值， 属于的图在batch里的编号


def extract_first_element_per_batch(tensor1, tensor2):
    # Get the unique batch indices from tensor2
    unique_batch_indices = torch.unique(tensor2)
    # Initialize a list to store the first elements of each batch item
    first_elements = []
    # Iterate through each unique batch index
    for batch_idx in unique_batch_indices:
        # Find the first occurrence of the batch index in tensor2
        first_occurrence = torch.nonzero(tensor2 == batch_idx, as_tuple=False)[0, 0]
        # Extract the first element from tensor1 and append it to the list
        first_element = tensor1[first_occurrence]
        first_elements.append(first_element)
    # Convert the list to a tensor
    result = torch.stack(first_elements, dim=0)
    return result


class LayerNorm(nn.Module):
    """
    layer normalization
    Simple layer norm object optionally used with the convolutional encoder.
    """

    def __init__(self, feature_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones((feature_dim,)))
        self.register_parameter("gamma", self.gamma)
        self.beta = nn.Parameter(torch.zeros((feature_dim,)))
        self.register_parameter("beta", self.beta)
        self.eps = eps

    def forward(self, x):
        # x: [batch_size, embed_dim]
        # normalize for each embedding
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # output shape is the same as x
        # Type not match for self.gamma and self.beta??????????????????????
        # output: [batch_size, embed_dim]
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def get_activation_function(activation, context_str):
    if activation == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise Exception("{} activation not recognized.".format(context_str))


class SingleFeedForwardNN(nn.Module):
    """
        Creates a single layer fully connected feed forward neural network.
        this will use non-linearity, layer normalization, dropout
        this is for the hidden layer, not the last layer of the feed forard NN
    """

    def __init__(self, input_dim,
                 output_dim,
                 dropout_rate=None,
                 activation="sigmoid",
                 use_layernormalize=False,
                 skip_connection=False,
                 context_str=''):
        '''
        Args:
            input_dim (int32): the input embedding dim
            output_dim (int32): dimension of the output of the network.
            dropout_rate (scalar tensor or float): Dropout keep prob.
            activation (string): tanh or relu or leakyrelu or sigmoid
            use_layernormalize (bool): do layer normalization or not
            skip_connection (bool): do skip connection or not
            context_str (string): indicate which spatial relation encoder is using the current FFN
        '''
        super(SingleFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if dropout_rate is not None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

        self.act = get_activation_function(activation, context_str)

        if use_layernormalize:
            # the layer normalization is only used in the hidden layer, not the last layer
            self.layernorm = nn.LayerNorm(self.output_dim)
        else:
            self.layernorm = None

        # the skip connection is only possible, if the input and out dimention is the same
        if self.input_dim == self.output_dim:
            self.skip_connection = skip_connection
        else:
            self.skip_connection = False

        self.linear = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size,..., output_dim]
            note there is no non-linearity applied to the output.
        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim
        # Linear layer
        output = self.linear(input_tensor)
        # non-linearity
        output = self.act(output)
        # dropout
        if self.dropout is not None:
            output = self.dropout(output)

        # skip connection
        if self.skip_connection:
            output = output + input_tensor

        # layer normalization
        if self.layernorm is not None:
            output = self.layernorm(output)

        return output


class MultiLayerFeedForwardNN(nn.Module):
    """
        Creates a fully connected feed forward neural network.
        N fully connected feed forward NN, each hidden layer will use non-linearity, layer normalization, dropout
        The last layer do not have any of these
    """

    def __init__(self, input_dim,
                 output_dim,
                 num_hidden_layers=0,
                 dropout_rate=0.1,
                 hidden_dim=-1,
                 activation="relu",
                 use_layernormalize=True,
                 skip_connection=False,
                 context_str=None):
        '''
        Args:
            input_dim (int32): the input embedding dim
            num_hidden_layers (int32): number of hidden layers in the network, set to 0 for a linear network.
            output_dim (int32): dimension of the output of the network.
            dropout (scalar tensor or float): Dropout keep prob.
            hidden_dim (int32): size of the hidden layers
            activation (string): tanh or relu
            use_layernormalize (bool): do layer normalization or not
            context_str (string): indicate which spatial relation encoder is using the current FFN
        '''
        super(MultiLayerFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_layernormalize = use_layernormalize
        self.skip_connection = skip_connection
        self.context_str = context_str

        self.layers = nn.ModuleList()
        if self.num_hidden_layers <= 0:
            self.layers.append(SingleFeedForwardNN(input_dim=self.input_dim,
                                                   output_dim=self.output_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=False,
                                                   skip_connection=False,
                                                   context_str=self.context_str))
        else:
            self.layers.append(SingleFeedForwardNN(input_dim=self.input_dim,
                                                   output_dim=self.hidden_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=self.use_layernormalize,
                                                   skip_connection=self.skip_connection,
                                                   context_str=self.context_str))

            for i in range(self.num_hidden_layers - 1):
                self.layers.append(SingleFeedForwardNN(input_dim=self.hidden_dim,
                                                       output_dim=self.hidden_dim,
                                                       dropout_rate=self.dropout_rate,
                                                       activation=self.activation,
                                                       use_layernormalize=self.use_layernormalize,
                                                       skip_connection=self.skip_connection,
                                                       context_str=self.context_str))

            self.layers.append(SingleFeedForwardNN(input_dim=self.hidden_dim,
                                                   output_dim=self.output_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=False,
                                                   skip_connection=False,
                                                   context_str=self.context_str))

    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size, ..., output_dim]
            note there is no non-linearity applied to the output.
        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim

        output = input_tensor
        for layer in self.layers:
            output = layer(output)

        return output


def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    freq_list = None
    if freq_init == "random":
        freq_list = torch.rand(frequency_num) * max_radius
    elif freq_init == "geometric":
        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) / (frequency_num * 1.0 - 1))
        timescales = min_radius * torch.exp(torch.arange(frequency_num, dtype=torch.float32) * log_timescale_increment)
        freq_list = 1.0 / timescales
    return freq_list


class GridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    """

    def __init__(self, spa_embed_dim, coord_dim=2, frequency_num=16,
                 max_radius=0.01, min_radius=0.00001,
                 freq_init="geometric",
                 ffn=None):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(GridCellSpatialRelationEncoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.ffn = ffn
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()
        self.input_embed_dim = self.cal_input_dim()

        if self.ffn is not None:
            self.ffn = MultiLayerFeedForwardNN(2 * frequency_num * 2, spa_embed_dim)

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = torch.unsqueeze(self.freq_list, 1)
        # self.freq_mat shape: (frequency_num, 2)
        self.freq_mat = freq_mat.repeat(1, 2)

    def make_input_embeds(self, coords):
        # coords: shape (batch_size, num_context_pt, 2)
        batch_size, num_context_pt, _ = coords.shape
        # coords: shape (batch_size, num_context_pt, 2, 1, 1)
        coords = coords.unsqueeze(-1).unsqueeze(-1)
        # coords: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        coords = coords.repeat(1, 1, 1, self.frequency_num, 2)
        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        spr_embeds = coords * self.freq_mat.to(self.device)
        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, 2*frequency_num*2=input_embed_dim)
        spr_embeds[:, :, :, :, 0::2] = torch.sin(spr_embeds[:, :, :, :, 0::2])  # dim 2i
        spr_embeds[:, :, :, :, 1::2] = torch.cos(spr_embeds[:, :, :, :, 1::2])  # dim 2i+1
        # (batch_size, num_context_pt, 2*frequency_num*2)
        spr_embeds = torch.reshape(spr_embeds, (batch_size, num_context_pt, -1))
        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_input_embeds(coords)
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb

class GAT_Unet(nn.Module):
    """
        GCN with positional encoder and auxiliary tasks
    """

    def __init__(self, T, tdim, cemb_dim, nheads, emb_dim, input_emb, emb_hidden_dim, num_features_in=1, num_features_out=1, k=20, conv_dim=64):
        super(GAT_Unet, self).__init__()
        self.dropout = 0
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emb_hidden_dim = emb_hidden_dim
        self.emb_dim = emb_dim
        self.cemb_dim = cemb_dim
        self.tdim = tdim
        self.k = k
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.spenc = GridCellSpatialRelationEncoder(
            spa_embed_dim=emb_hidden_dim, ffn=True, min_radius=1e-06, max_radius=360
        )
        self.dec = nn.Sequential(
            nn.Linear(emb_hidden_dim, emb_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(emb_hidden_dim // 2, emb_hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(emb_hidden_dim // 4, emb_dim)
        )
       
        self.time_embedding = TimeEmbedding(T, 32, tdim)

        self.valuelayer = nn.Linear(num_features_in, input_emb)
        
        
        self.attentions = [GraphAttentionLayer(input_emb + emb_dim + tdim + cemb_dim+1,
                                               emb_hidden_dim,
                                               dropout=self.dropout,
                                               alpha=0.1,
                                               concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.mid_att = [GraphAttentionLayer(emb_hidden_dim * nheads,
                                            emb_hidden_dim // 2,
                                            dropout=self.dropout,
                                            alpha=0.1,
                                            concat=True) for _ in range(nheads//2)]
        
        for i, attention in enumerate(self.mid_att):
            self.add_module('attention_{}'.format(i), attention)
            
        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=emb_hidden_dim*nheads)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=emb_hidden_dim*nheads // 4)
        self.out_att = GraphAttentionLayer((emb_hidden_dim // 2) * (nheads//2),       # emb_hidden_dim * nhead
                                           num_features_in,
                                           dropout=self.dropout,
                                           alpha=0.1,
                                           concat=False)
        self.mid_att = nn.ModuleList(self.mid_att)
        self.attentions = nn.ModuleList(self.attentions)
        # init weights
        for p in self.dec.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)

        self.reconlayer_in = nn.Linear(1,64)
        self.reconlayer_out = nn.Linear(64,1)

        
        self.tif_conv = nn.Conv2d(3, 1, kernel_size=11, padding=5)
        init.xavier_normal_(self.tif_conv.weight)
        self.tif_conv = self.tif_conv.to(self.device)
        
    def tif_emb(self, tif, coords):
        tif_conv = self.tif_conv(tif).squeeze()
        # tif_point = tif_conv.view(tif_conv.shape[0], 1, 1).expand((coords.shape[0],coords.shape[1], 1))
        # print(tif_conv.shape)
        tif_point = tif_conv[
            torch.arange(tif.shape[0]).view(-1, 1, 1).long(),
            coords[:, :, 0].unsqueeze(2).long(),
            coords[:, :, 1].unsqueeze(2).long()
        ]
        return tif_point
        
    def forward(self, inputs, targets, coords, input_lenths, condition, T, tif):  # input[batch, len, 10]
        # recon_x = self.reconlayer_out(self.reconlayer_in(inputs)) 
        
        emb = self.spenc(coords)
        emb = self.dec(emb)

        emb_t = self.time_embedding(T)

        tif_point = self.tif_emb(tif, coords)
        # tif_point = torch.randn(*inputs)
        
        emb_l, indexer, mask = padded_seq_to_vectors(emb, input_lenths)
        count_per_batch = torch.bincount(indexer)

        gnn_condition = condition.repeat_interleave(count_per_batch, dim=0)
        gnn_t = emb_t.repeat_interleave(count_per_batch, dim=0)
        # print('indexer, ', indexer.shape)
        # print(count_per_batch)
        
        # gnn_condition, _ = padded_seq_to_vectors(condition, input_lenths)
        lur, _, _ = padded_seq_to_vectors(tif_point, input_lenths)
        x_l, _, _ = padded_seq_to_vectors(inputs, input_lenths)  # x_l[len, 10]
        x_l = self.valuelayer(x_l)
        # lur = self.valuelayer(lur)
        # print(lur.shape)
        # print(x_l.shape)
        
        # print('1, ', x_l.shape)
        # print('2, ', emb_l.shape)
        # print('3, ', gnn_t.shape)
        # print('4, ', gnn_condition.shape)
        # x_l = x_l[:, 0:1]
        y_l, _, _ = padded_seq_to_vectors(targets, input_lenths)
        c_l, _, _ = padded_seq_to_vectors(coords, input_lenths)
      
        edge_index = knn_graph(c_l, k=self.k, batch=indexer)
        edge_weight = makeEdgeWeight(c_l, edge_index).to(self.device)
        adj = to_dense_adj(edge_index).squeeze(0)

        x = torch.cat((x_l, emb_l, gnn_t, gnn_condition, lur), dim=1)  # emb_l[len, 16], gnn_t[len. 128], condition[len, 32]
        # print(x.shape)
        # x = torch.cat((x_l, emb_l, gnn_t), dim=1)
        # print("x before g: ", torch.isnan(x).any())
        g = F.dropout(x, self.dropout, training=self.training)
        # print(torch.isnan(g).any())
        # print("g before att: ", torch.isnan(g).any())
        g1 = torch.cat([att(g, adj) for att in self.attentions], dim=1)
        # print("g after att: ", torch.isnan(g).any())
        g2 = self.layernorm1(g1)
        g3 = F.dropout(g2, self.dropout, training=self.training)
        g4 = torch.cat([att(g3, adj) for att in self.mid_att], dim=1)
        g5 = self.layernorm2(g4)
        # if  torch.isnan(g2).any() == True:
            # print("condition: ", torch.isnan(condition).any())
            # print("count_per_batch: ", torch.isnan(count_per_batch).any())
            # print("indexer: ", torch.isnan(indexer).any())
            # print("emb_l: ", torch.isnan(emb_l).any())
            # print("gnn_t: ", torch.isnan(gnn_t).any())
            # print("gnn_condition: ", torch.isnan(gnn_condition).any())
            # print("x_l: ", torch.isnan(x_l).any())
            # print("x: ", torch.isnan(x).any())
            # print("g before att: ", torch.isnan(g).any())
            # print("g1 after att: ", torch.isnan(g1).any())
            # print("g2 after att+drop: ", torch.isnan(g2).any())
        # print("g before out att: ", torch.isnan(g).any())
        g6 = F.dropout(g5, self.dropout, training=self.training)
        g7 = self.leakyrelu(self.out_att(g6, adj))
        # print("g after out att: ", torch.isnan(g).any())
        # print('g7: ', g7.shape)

        batch_size = inputs.shape[0]
        max_len = inputs.shape[1]
        
        output_list = torch.split(g7, torch.bincount(indexer.squeeze()).tolist())
        output_list_full = pad_sequence(output_list, batch_first=True, padding_value=-1)
        # print(output_list_full.shape)
        padded_output = -1 * torch.ones(batch_size, max_len, 1)
        padded_output[:, :output_list_full.shape[1], :] = output_list_full
        # mask = (output_list_full != -1).float()
        target_list = torch.split(y_l, torch.bincount(indexer.squeeze()).tolist())
        target_list_full = pad_sequence(target_list, batch_first=True, padding_value=-1)
        padded_target = -1 * torch.ones(batch_size, max_len, 1)
        padded_target[:, :target_list_full.shape[1], :] = target_list_full
        
        coord_list = torch.split(c_l, torch.bincount(indexer.squeeze()).tolist())
        coord_list_full = pad_sequence(coord_list, batch_first=True, padding_value=-1)
        padded_coord = -1 * torch.ones(batch_size, max_len, 2)
        padded_coord[:, :coord_list_full.shape[1], :] = coord_list_full

        mask = mask.unsqueeze(-1)
        
        return padded_output.to(self.device), padded_target.to(self.device), padded_coord.to(self.device), mask.to(self.device)
     
