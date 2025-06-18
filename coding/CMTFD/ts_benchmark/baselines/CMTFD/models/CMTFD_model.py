from ts_benchmark.baselines.CMTFD.layers.linear_extractor_cluster import Linear_extractor_cluster
import torch.nn as nn
from einops import rearrange
from ts_benchmark.baselines.CMTFD.utils.masked_attention import Mahalanobis_mask, Encoder, EncoderLayer,AttentionLayer,FullAttention
import torch

import torch.nn.functional as F
from einops import rearrange


class SimpleBandFilter(nn.Module):
    def __init__(self, keep='low', ratio=0.3):
        super().__init__()
        self.keep = keep  # 'low' 或 'high'
        self.ratio = ratio

    def forward(self, x):
        # x: [B, N, L]
        freqs = torch.fft.rfft(x, dim=-1, norm='ortho')
        B, N, Lf = freqs.shape
        keep_len = int(Lf * self.ratio)

        if self.keep == 'low':
            mask = torch.zeros_like(freqs)
            mask[:, :, :keep_len] = 1
        else:
            mask = torch.zeros_like(freqs)
            mask[:, :, -keep_len:] = 1

        filtered_freqs = freqs * mask
        return torch.fft.irfft(filtered_freqs, n=x.shape[-1], dim=-1, norm='ortho')


class CMTFDModel(nn.Module):
    def __init__(self, config):
        super(CMTFDModel, self).__init__()
        self.config = config
        self.cluster = Linear_extractor_cluster(config)
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.seq_len = config.seq_len
        self.d_model = config.d_model
        self.mask_generator = Mahalanobis_mask(config.seq_len)  # 通道掩码生成
        self.Channel_transformer = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(
                        mask_flag=True,
                        factor=config.factor,
                        attention_dropout=config.dropout,
                        output_attention=config.output_attention
                    ),
                    config.d_model,
                    config.n_heads
                ),
                config.d_model,
                config.d_ff,
                dropout=config.dropout,
                activation=config.activation
            ) for _ in range(config.e_layers)
        ], norm_layer=torch.nn.LayerNorm(config.d_model))

        self.linear_head = nn.Sequential(
            nn.Linear(config.d_model, config.pred_len),
            nn.Dropout(config.fc_dropout)
        )

        self.band_filter = SimpleBandFilter(keep='low', ratio=0.4)

    def forward(self, input):
        if input.dim() != 3:
            raise ValueError(f"Expected input shape [B, L, N], but got {input.shape}")

        device = input.device

        if self.CI:
            channel_independent_input = rearrange(input, 'b l n -> (b n) l 1')
            reshaped_output, L_importance = self.cluster(channel_independent_input)
            temporal_feature = rearrange(reshaped_output, '(b n) l 1 -> b l n', b=input.shape[0])
        else:
            temporal_feature, L_importance = self.cluster(input)

        temporal_feature = rearrange(temporal_feature, 'b l n -> b n l')
        if hasattr(self.cluster, 'proj'):
            temporal_feature = self.cluster.proj(temporal_feature)

        if temporal_feature.is_complex():
            temporal_feature = temporal_feature.real

        # 使用 band_filter 进行频域滤波
        freq_enhanced = self.band_filter(temporal_feature)

        # 生成通道掩码并计算对应的对比损失
        if self.n_vars > 1:
            changed_input = rearrange(input, 'b l n -> b n l')
            channel_mask = self.mask_generator(changed_input)
            if self.training:
                # 获取通道掩码的对比损失
                channel_contrast_loss = getattr(self.mask_generator, 'contrast_loss', torch.tensor(0.0, device=device))
            else:
                channel_contrast_loss = torch.tensor(0.0, device=device)
            # 使用通道掩码进行transform
            channel_group_feature, _ = self.Channel_transformer(x=freq_enhanced, attn_mask=channel_mask)
            output = self.linear_head(channel_group_feature)
        else:
            output = self.linear_head(freq_enhanced)
            channel_contrast_loss = torch.tensor(0.0, device=device)

        output = rearrange(output, 'b n d -> b d n')
        if hasattr(self.cluster, 'revin'):
            output = self.cluster.revin(output, "denorm")

        # 确保所有返回值都是张量
        if not isinstance(L_importance, torch.Tensor):
            L_importance = torch.tensor(L_importance, dtype=torch.float32, device=device)
        if not isinstance(channel_contrast_loss, torch.Tensor):
            channel_contrast_loss = torch.tensor(channel_contrast_loss, dtype=torch.float32, device=device)

        # 返回一个固定的损失值作为adv_loss
        adv_loss = torch.tensor(0.0, device=device)

        return output, L_importance, channel_contrast_loss, adv_loss