import torch
import torch.nn as nn

import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, start_pos=0, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(0)
        self.register_buffer('pe', pe)
        self.start_pos = start_pos

    def forward(self, x):
        pos = self.pe[:, :, self.start_pos:x.size(2) + self.start_pos, :]
        return pos


class TokenEmbedding(nn.Module):
    def __init__(self, channel, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv2d(in_channels=channel, out_channels=d_model, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.tokenConv(x).permute(0, 3, 2, 1)
        return x  # B, V, N, D


class DataEmbedding(nn.Module):
    def __init__(self, patch, d_model, start_pos=0):
        super(DataEmbedding, self).__init__()
        self.patch = patch
        self.value_embedding = TokenEmbedding(channel=patch, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model, start_pos=start_pos)

    def forward(self, x):
        B, L, V = x.shape
        x = x.contiguous().view(B, -1, self.patch, V).transpose(1, 2)  # B, P, N, V
        pos = self.position_embedding(x)
        x = self.value_embedding(x) + pos  # B, V, N, D
        return x
