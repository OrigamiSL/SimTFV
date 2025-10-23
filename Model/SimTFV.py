# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
from Model.Modules import *
from Model.embed import *
from utils.RevIN import RevSTIN


class Encoder_process(nn.Module):
    def __init__(self, short_input_len, encoder_layers, patch_size, d_model, dropout, Not_use_CV):
        super(Encoder_process, self).__init__()
        self.patch_size = patch_size
        self.Embed = DataEmbedding(patch_size, d_model)
        self.encoder_layers = np.cumsum(np.array(encoder_layers).reshape(-1), axis=0)[:-1].tolist()
        self.encoders = []
        for i, el in enumerate(encoder_layers):
            self.encoders += [Encoder(short_input_len // (patch_size * 2 ** i), d_model * 2 ** i, dropout, Not_use_CV)
                              for _ in range(el)]
        self.encoders = nn.ModuleList(self.encoders)
        self.d_model = d_model

    def forward(self, x):
        B, _, V = x.shape
        x_enc = self.Embed(x)

        x_enc_list = []
        last_patch = 0
        for i, encoder in enumerate(self.encoders):
            x_enc, CV_PATCH = encoder(x_enc)
            if (i + 1) in self.encoder_layers:
                B, V, N, D = x_enc.shape
                # x_enc_list.append(x_enc[:, :, -CV_PATCH:, :])
                x_enc_list.append(x_enc)
                x_enc = x_enc.contiguous().view(B, V, N // 2, 2, D).view(B, V, N // 2, 2 * D)
            last_patch = CV_PATCH
        # x_enc_list.append(x_enc[:, :, -last_patch:, :])
        x_enc_list.append(x_enc)
        return x_enc_list


class Decoder_process(nn.Module):
    def __init__(self, input_len, decoder_layers, patch_size, d_model, dropout):
        super(Decoder_process, self).__init__()
        self.d_model = d_model
        self.decoder_layers = np.cumsum(np.array(decoder_layers).reshape(-1), axis=0)[:-1].tolist()
        self.Embed = DataEmbedding(patch_size, d_model, start_pos=input_len // patch_size)
        self.decoders = []
        for i, dl in enumerate(decoder_layers):
            self.decoders += [Decoder(d_model * 2 ** i, dropout)
                              for _ in range(dl)]
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x_enc_list, y):
        B, _, V = y.shape
        y_dec = self.Embed(y)  # B, V, N, D

        j = 0
        y_dec_list = []
        for i, decoder in enumerate(self.decoders):
            y_dec = decoder(x_enc_list[j], y_dec)
            if (i + 1) in self.decoder_layers:
                B, V, N, D = y_dec.shape
                y_dec_list.append(y_dec)
                y_dec = y_dec.contiguous().view(B, V, N // 2, 2, D).view(B, V, N // 2, 2 * D)
                j += 1
        y_dec_list.append(y_dec)
        return y_dec_list


class Model(nn.Module):
    def __init__(self, long_input_len, short_input_len, pred_len,
                 encoder_layers, decoder_layers, patch_size, d_model, dropout, decoder_IN, Not_use_CV):
        super(Model, self).__init__()
        self.input_len = long_input_len
        self.pred_len = pred_len
        self.patch_size = patch_size
        self.encoder_layers = encoder_layers
        self.d_model = d_model

        self.revstin = RevSTIN(short_input_len)
        assert (len(encoder_layers) == len(decoder_layers))
        self.Encoder_process = (
            Encoder_process(short_input_len, encoder_layers, patch_size, d_model, dropout, Not_use_CV))
        self.Decoder_process = (
            Decoder_process(long_input_len, decoder_layers, patch_size,
                            d_model, dropout))
        self.dlinear = [nn.Linear(d_model * 2 ** i, patch_size * 2 ** i) for i in range(len(decoder_layers))]
        self.dlinear = nn.ModuleList(self.dlinear)
        self.DIN = decoder_IN

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        B, _, V = x.shape
        self.revstin(x, 'stats')
        x_enc = self.revstin(x, 'norm')
        x_enc_list = self.Encoder_process(x_enc)
        y = torch.zeros([B, self.pred_len, V]).to(x.device)
        if self.DIN:
            y_dec = self.revstin(y, 'norm')
        else:
            y_dec = y
        y_dec_list = self.Decoder_process(x_enc_list, y_dec)
        y_out = 0
        for i, (cy_dec, dlinear) in enumerate(zip(y_dec_list, self.dlinear)):
            y_out += dlinear(cy_dec).contiguous().view(B, V, -1).transpose(1, 2)

        out = self.revstin(y_out, 'denorm')
        return out
