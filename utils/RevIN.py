import torch
import torch.nn as nn
import math


def my_kl(p, q):  # K-L divergence
    res1 = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    res2 = q * (torch.log(q + 0.0001) - torch.log(p + 0.0001))
    return torch.sum(res1, dim=-1) + torch.sum(res2, dim=-1)


class RevSTIN(nn.Module):
    def __init__(self, segment_len=336, eps=1e-5):
        """
        :param eps: a value added for numerical stability
        """
        super(RevSTIN, self).__init__()
        self.seg_len = segment_len
        self.norm = nn.LayerNorm(segment_len, elementwise_affine=False)
        self.eps = eps

    def forward(self, x, mode: str):
        if mode == 'stats':
            self._get_statistics(x)
        elif mode == 'norm':
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        B, L, V = x.shape
        x_seg = x.unfold(dimension=1, size=self.seg_len, step=1)  # B, N, V, P
        x_seg_norm = self.norm(x_seg)
        x_seg_fft = torch.abs(torch.fft.fft(x_seg_norm, dim=-1)) + 0.0001
        x_seg_fft_norm = x_seg_fft / torch.sum(x_seg_fft, dim=-1, keepdim=True)
        x_uni = torch.ones_like(x_seg_fft_norm) / self.seg_len
        x_dis = my_kl(x_seg_fft_norm, x_uni)
        IN_weight = torch.softmax(x_dis, dim=1)  # B, N, V

        self.mean = torch.sum(torch.mean(x_seg, dim=-1) * IN_weight, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean[..., :x.shape[-1]]
        x = x / self.stdev[..., :x.shape[-1]]
        return x

    def _denormalize(self, x):
        x = x * self.stdev[..., :x.shape[-1]]
        x = x + self.mean[..., :x.shape[-1]]
        return x
