import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class Mask:
    def __init__(self, B, N, V, device="cpu"):
        with torch.no_grad():
            _mask = torch.ones([V, V], dtype=torch.bool).to(device)
            self._mask = _mask.unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)

    @property
    def mask(self):
        return self._mask


class Attn_FullLevel(nn.Module):
    def __init__(self, CV_patch_num, d_model, dropout=0.1, Not_use_CV=False):
        super(Attn_FullLevel, self).__init__()
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection1 = nn.Linear(d_model, d_model)
        self.key_projection2 = nn.Linear(d_model, d_model)
        self.value_projection1 = nn.Linear(d_model, d_model)
        self.value_projection2 = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.CV_patch_num = CV_patch_num
        self.Not_use_CV = Not_use_CV

    def forward(self, queries, keys, values):
        B, V, N, D = queries.shape
        _, _, S, _ = keys.shape
        scale = 1. / math.sqrt(D)

        queries = self.query_projection(queries)
        keys1 = self.key_projection1(keys)
        values1 = self.value_projection1(values)

        assert (N == S)
        scoresT = torch.einsum("bvnd,bvsd->bvns", queries[:, :, -self.CV_patch_num:], keys1)  # [B V Ncv N]
        if not self.Not_use_CV:
            keys2 = self.key_projection2(keys)
            values2 = self.value_projection2(values)
            scoresV = torch.einsum("bvnd,bvsd->bvns", queries[:, :, -self.CV_patch_num:].transpose(1, 2),
                                   keys2[:, :, -self.CV_patch_num:].transpose(1, 2))  # [B Ncv V V]
            scores = torch.cat([scoresT, scoresV.transpose(1, 2)], dim=-1)  # [B V Ncv N+V]
        else:
            scores = scoresT

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        if not self.Not_use_CV:
            attnT, attnV = attn[..., :N], attn[..., -V:].transpose(1, 2)
            outT = torch.einsum("bvns,bvsd->bvnd", attnT, values1)  # [B V Ncv D]
            outV = torch.einsum("bnvs,bnsd->bnvd", attnV[:, -self.CV_patch_num:],
                                values2[:, :, -self.CV_patch_num:].transpose(1, 2))  # [B Nc V D]
            outCV = outT + outV.transpose(1, 2)
        else:
            outCV = torch.einsum("bvns,bvsd->bvnd", attn, values1)  # [B V Ncv D]

        if self.CV_patch_num != N:
            out = torch.cat([values1[:, :, :-self.CV_patch_num], outCV], dim=2)
        else:
            out = outCV
        return self.out_projection(out), self.CV_patch_num  # [B V N D]


class Attn_TemporalLevel(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Attn_TemporalLevel, self).__init__()
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        B, V, N, D = queries.shape
        _, V, M, _ = keys.shape
        scale = 1. / math.sqrt(D)

        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        scores = torch.einsum("bvnd,bvmd->bvnm", queries, keys)  # [B V N M]
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bvnm,bvmd->bvnd", attn, values)  # [B V N D]
        return self.out_projection(out)  # [B V N D]


class Encoder(nn.Module):
    def __init__(self, CV_patch_num, d_model, dropout=0.1, Not_use_CV=False):
        super(Encoder, self).__init__()
        self.patch_dim = d_model
        self.attn = Attn_FullLevel(CV_patch_num, d_model, dropout, Not_use_CV)

        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(self.patch_dim)
        self.norm2 = nn.LayerNorm(self.patch_dim)

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.patch_dim, 4 * self.patch_dim)
        self.linear2 = nn.Linear(4 * self.patch_dim, self.patch_dim)

    def forward(self, x):
        attn_x, CV_PATCH = self.attn(x, x, x)
        z = x = self.norm1(x + self.dropout(attn_x))
        z = self.activation(self.linear1(z))
        x = self.norm2(x + self.dropout(self.linear2(z)))  # B, VN, D
        return x, CV_PATCH


class Decoder(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Decoder, self).__init__()
        self.patch_dim = d_model
        self.attn = Attn_TemporalLevel(d_model, dropout)

        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(self.patch_dim)
        self.norm2 = nn.LayerNorm(self.patch_dim)

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.patch_dim, 4 * self.patch_dim)
        self.linear2 = nn.Linear(4 * self.patch_dim, self.patch_dim)

    def forward(self, x, y):
        attn_y = self.attn(y, x, x)
        z = y = self.norm1(y + self.dropout(attn_y))
        z = self.activation(self.linear1(z))
        y = self.norm2(y + self.dropout(self.linear2(z)))  # B, V, N, D
        return y
