import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.cuda.amp import autocast
from einops import rearrange, repeat
import torch.nn.init as init

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


def padded_seq_to_vectors(padded_seq, logger):
    # Get the actual lengths of each sequence in the batch
    actual_lengths = logger.int()
    # Step 1: Form the first tensor containing all actual elements from the batch
    mask = torch.arange(padded_seq.size(1), device=padded_seq.device) < actual_lengths.view(-1, 1)
    tensor1 = torch.masked_select(padded_seq, mask.unsqueeze(-1)).view(-1, padded_seq.size(-1))
    # Step 2: Form the second tensor to record which row each element comes from
    tensor2 = torch.repeat_interleave(torch.arange(padded_seq.size(0), device=padded_seq.device), actual_lengths)
    return tensor1, tensor2  # 真实值， 属于的图在batch里的编号

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
        self.cls_token = nn.Parameter(torch.empty((12,), device=self.device))
        init.kaiming_uniform_(self.cls_token.unsqueeze(0))
        self.cls_token.requires_grad = True
        self.cls_token.squeeze(0)
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
    def forward(self, input_lenths, input_series):
        # generate & attach class token
        #     - 0 bit: Sensor reading: learnable
        #     - 1 bit: Sensor class: from Q_tokens
        #     - 2~3 bit: Intp Location: from Q_tokens
        #     - 4~12 bit: Intp Ob Property: 'mcpm10' (001000000)
        
        # x_l, _ = padded_seq_to_vectors(input_series, input_lenths)
        
        cls_token_reading = self.cls_token.repeat(len(input_lenths), 1).to(self.device).unsqueeze(dim=1)
        input_series = torch.concat([cls_token_reading, input_series], dim=1)

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
        # print("x: ", torch.isnan(x).any())
        hidden_states = self.transformer(x, src_key_padding_mask=attention_mask)
        # print("hidden_states: ", torch.isnan(hidden_states).any())
        mlp_inputs = hidden_states[:, 0, :]
        # print("mlp_inputs: ", torch.isnan(mlp_inputs).any())
        # go pass MLP
        mu = self.fc_mu(mlp_inputs)
        log_var = self.fc_var(mlp_inputs)
        z = self.reparameterize(mu, log_var)
        
        # output = self.output(hidden_states)
        
        return mlp_inputs  # output, mu, log_var
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

        'batch': 4,
        'accumulation_steps': 256 // 128,
        'epoch': 100,
        'trans_lr': 1e-7,
        'nn_lr': 1e-6,
        'es_mindelta': 1.0,
        'es_endure': 50,

        'word_col': [0, 1],
        'pos_col': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
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
    model = Transformer(settings, device)
    q = torch.randn(batch_size, 12)
    x = torch.randn(batch_size, 240, 12)
    l = torch.tensor([240, 240])

    y = model(l.to(device),x.to(device))

    # print(y.shape)
