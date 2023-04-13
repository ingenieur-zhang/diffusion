import torch
import torch.nn as nn
import zipfile
import pickle
import numpy as np
import torch.nn.functional as F
from osgeo import gdal
import codecs
import csv
import pandas as pd
import os
import copy
from attention import MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward, Embeddings, Encoder, EncoderLayer, FullEncoder

has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')

in_path = './Input_Approx_Origin/'
in_zipname = 'Input_Approx.zip'

dir_path = in_path

def make_model(
    src_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = FullEncoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),

    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
