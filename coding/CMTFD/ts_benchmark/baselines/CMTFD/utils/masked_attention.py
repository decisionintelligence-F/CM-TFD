import torch.nn as nn
import torch
from math import sqrt

import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
import math
import torch.fft
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F



class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # if self.mask_flag:
        #     large_negative = -math.log(1e10)
        #     attention_mask = torch.where(attn_mask == 0, torch.tensor(large_negative), attn_mask)
        #
        #     scores = scores * attention_mask
        if self.mask_flag:
            large_negative = -math.log(1e10)
            attention_mask = torch.where(attn_mask == 0, large_negative, 0)

            scores = scores * attn_mask + attention_mask

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out + queries
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


import numpy as np  
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Mahalanobis_mask(nn.Module):
    def __init__(self, input_size):
        super(Mahalanobis_mask, self).__init__()
        self.wavelets = ['db4', 'sym5', 'coif5', 'bior2.4']
        self.level = 4

        self.weights = nn.Parameter(torch.ones(self.level + 1))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.wavelet_weights = nn.Parameter(torch.ones(len(self.wavelets)))

        self.raw_pos_thresh = nn.Parameter(torch.tensor(0.0))
        self.raw_neg_gap = nn.Parameter(torch.tensor(-0.5))

        self.min_gap = 0.05
        self.A = None

        # contrast loss for adversarial mask
        self.contrast_loss = torch.tensor(0.0)

    def get_thresholds(self):
        pos_thresh = torch.sigmoid(self.raw_pos_thresh)
        neg_thresh = pos_thresh - torch.sigmoid(self.raw_neg_gap) * pos_thresh
        return pos_thresh, neg_thresh

    def multi_wavelet_reconstruct(self, X):
        all_recon = []
        for wavelet in self.wavelets:
            coeffs = pywt.wavedec(X.cpu().numpy(), wavelet, level=self.level, axis=-1)
            reconstructed = []
            for i in range(len(coeffs)):
                coeffs_temp = [np.zeros_like(c) for c in coeffs]
                coeffs_temp[i] = coeffs[i]
                recon = pywt.waverec(coeffs_temp, wavelet)
                recon = torch.tensor(recon, device=X.device).float()
                recon = F.interpolate(recon, size=X.shape[-1], mode='linear')
                reconstructed.append(recon)
            all_recon.append(torch.stack(reconstructed, dim=1))

        wavelet_weights = F.softmax(self.wavelet_weights, dim=0)
        return torch.einsum('w,blwct->blct', wavelet_weights, torch.stack(all_recon, dim=2))

    def compute_channel_affinity(self, X, mode='pos'):
        X_recon = self.multi_wavelet_reconstruct(X)
        B, L, C, T = X_recon.shape
        energy = torch.norm(X_recon, dim=-1, keepdim=True)
        band_weights = F.softmax(energy, dim=1)
        X_fused = torch.sum(X_recon * band_weights, dim=1)

        if self.A is None:
            dim = X_fused.shape[-1]
            self.A = nn.Parameter(torch.randn(dim, dim))
            nn.init.orthogonal_(self.A)

        # 🔧 确保 A 在和 X 同一个 device 上
        self.A.data = self.A.data.to(X.device)

        if mode == 'neg':
            noise = torch.randn_like(X_fused) * 0.01
            X_fused = X_fused + noise

        X1 = X_fused.unsqueeze(2)
        X2 = X_fused.unsqueeze(1)
        diff = X1 - X2
        temp = torch.einsum("dk,bxck->bxcd", self.A, diff)
        dist = torch.einsum("bxcd,bxcd->bxc", temp, temp)
        exp_dist = torch.exp(-dist / self.temperature.clamp(min=0.1))
        exp_dist = exp_dist * (1 - torch.eye(exp_dist.shape[-1]).to(X.device))
        p = exp_dist / (exp_dist.sum(dim=-1, keepdim=True) + 1e-10)
        return p


    def adversarial_mask_loss(self, p_pos, p_neg):
        # KL散度损失 + 熵正则
        kl_loss = F.kl_div((p_pos + 1e-10).log(), p_neg + 1e-10, reduction='batchmean') + \
                  F.kl_div((p_neg + 1e-10).log(), p_pos + 1e-10, reduction='batchmean')
        entropy_reg = - (p_pos * (p_pos + 1e-10).log()).sum(dim=-1).mean()
        # 最小间隔损失
        pos_thresh, neg_thresh = self.get_thresholds()
        gap_loss = F.relu(self.min_gap - (pos_thresh - neg_thresh))

        return kl_loss + 0.1 * entropy_reg + 0.3 * gap_loss
    def forward(self, X):
        # 输入: [B, C, T]
        p_pos = self.compute_channel_affinity(X, mode='pos')
        p_neg = self.compute_channel_affinity(X, mode='neg')

        if self.training:
            self.contrast_loss = self.adversarial_mask_loss(p_pos, p_neg)

        pos_thresh, _ = self.get_thresholds()
        return (p_pos > pos_thresh).float().unsqueeze(1)  # [B,1,C,C]