import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.cuda.amp import autocast
from einops import rearrange, repeat


def length_to_mask(lengths, total_len, device):
    max_len = total_len
    mask = torch.arange(max_len).expand(lengths.shape[0], max_len).to(device) < lengths.unsqueeze(1)
    return mask


# Returns the closest number that is a power of 2 to the given real number x
def closest_power_of_2(x):
    return 2 ** round(math.log2(x))


# Returns a list of n numbers that are evenly spaced between a and b.
def evenly_spaced_numbers(a, b, n):
    if n == 1:
        return [(a+b)/2]
    step = (b-a)/(n-1)
    return [a + i*step for i in range(n)]
    
    
# generate a V-shape MLP as torch.nn.Sequential given input_size, output_size, and layer_count(only linear layer counted)
def generate_sequential(a, b, n):
    layer_sizes = evenly_spaced_numbers(a, b, n)
    layer_sizes = [int(layer_sizes[0])] + [int(closest_power_of_2(x)) for x in layer_sizes[1:-1]] + [int(layer_sizes[-1])]
    
    layers = []
    for i in range(n-1):
        layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        # if i == 0:
        #     layers.append(nn.Dropout(0.1))
        if i < n-2:
            layers.append(torch.nn.LeakyReLU(0.1))
            # layers.append(torch.nn.BatchNorm1d(layer_sizes[i+1]))
    
    model = torch.nn.Sequential(*layers)
    return model


class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class Transformer(nn.Module):
    def __init__(self, settings, device):
        super(Transformer, self).__init__()
        
        self.settings = settings
        self.device = device
        # word embedding layer
        self.s_embedding_layer = torch.nn.Linear(len(settings['word_col']), settings['embedding_dim']).to(self.device)
        # position embedding layer
        self.p_embedding_layer = torch.nn.Linear(len(settings['pos_col']), settings['embedding_dim']).to(self.device)
        # class token
        # self.cls_token = nn.Parameter(torch.randn((1, 1, 1), requires_grad=True, device=self.device))
        self.cls_token = nn.Parameter(torch.randn((3, ), requires_grad=True, device=self.device))  # 可学习参数
        # embedding_drop
        # self.embedding_drop = nn.Dropout(0.05)

        # Transformer Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            settings['embedding_dim'], settings['num_head'], settings['feedforward_dim'], settings['dropout'], NewGELU(), batch_first=True
        ).to(self.device)
        self.transformer = nn.TransformerEncoder(encoder_layer, settings['num_layers']).to(self.device)

        # 输出层
        self.fc_mu = generate_sequential(settings['embedding_dim'], settings['vi_dim'], settings['output_hidden_layers']+2).to(self.device)
        self.fc_var = generate_sequential(settings['embedding_dim'], settings['vi_dim'], settings['output_hidden_layers']+2).to(self.device)
        self.output = torch.nn.Linear(settings['vi_dim'], 1).to(self.device)
        
        # init weights
        torch.nn.init.kaiming_normal_(self.s_embedding_layer.weight)
        torch.nn.init.kaiming_normal_(self.p_embedding_layer.weight)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
        for p in self.fc_mu.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)
        for p in self.fc_var.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)
        torch.nn.init.kaiming_normal_(self.output.weight)
                
                
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        

    @autocast()
    def forward(self, q_tokens, input_lenths, input_series):
        # generate & attach class token
        #     - 0 bit: Sensor reading: learnable
        #     - 1 bit: Sensor class: from Q_tokens
        #     - 2~3 bit: Intp Location: from Q_tokens
        #     - 4~12 bit: Intp Ob Property: 'mcpm10' (001000000)
        cls_token_reading = repeat(self.cls_token, 'd -> b 1 d', b=len(input_lenths)).to(self.device)
        # input_series = input_series.unsqueeze(1)
        cls_token_onehot = repeat(torch.tensor([1, 0, 0, 1, 0, 0, 0, 0, 0, 0]), 'd -> b 1 d', b=len(input_lenths)).to(self.device)
        # print('q_tokens, ', q_tokens.shape)
        # print('cls_token_reading, ', cls_token_reading.shape)
        # print('cls_token_onehot, ', cls_token_onehot.shape)
        # print('input_series, ', input_series.shape)
        cls_tokens = torch.concat([cls_token_reading, cls_token_onehot], dim=2)
        # print(input_series)
        # print('cls_tokens, ', cls_tokens.shape)
        input_series = torch.concat([cls_tokens, q_tokens, input_series], dim=1)
        
        # get s_embedding input
        s_embedding_input = input_series[:, :, self.settings['word_col']].reshape((-1, len(self.settings['word_col'])))
        s_embedded = self.s_embedding_layer(s_embedding_input).reshape((len(input_lenths), -1, self.settings['embedding_dim']))
        
        # get pos embedding input
        p_embedding_input = input_series[:, :, self.settings['pos_col']].reshape((-1, len(self.settings['pos_col'])))
        p_embedded = self.p_embedding_layer(p_embedding_input).reshape((len(input_lenths), -1, self.settings['embedding_dim']))
        
        # combine embeddings
        x = s_embedded + p_embedded
        # x = self.embedding_drop(x)
        
        # go pass transformer
        total_length = input_series.size(1)
        attention_mask = length_to_mask(input_lenths+1, total_length, self.device) == False
        hidden_states = self.transformer(x, src_key_padding_mask=attention_mask)
        mlp_inputs = hidden_states[:, 0, :]
        
        # go pass MLP
        mu = self.fc_mu(mlp_inputs)
        log_var = self.fc_var(mlp_inputs)
        z = self.reparameterize(mu, log_var)
        
        output = self.output(z)
        
        return mlp_inputs
        