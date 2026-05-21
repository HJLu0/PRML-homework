import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SinusoidalPosEncoding, get_pos_encoding

# CNN 编码器层（单层：Conv + GLU + 残差）


class CNNEncoderLayer(nn.Module):

    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 输出 2*d_model，一半作 gate，一半作内容
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=2 * d_model,
            kernel_size=kernel_size,
            padding=padding
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq, d_model)
        residual = x
        x = self.dropout(x)
        # Conv1d 需要 (batch, channels, seq)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)          # → (batch, seq, 2*d_model)
        # GLU: 将通道分两半
        x = F.glu(x, dim=-1)           # → (batch, seq, d_model)
        # 残差 + LayerNorm
        x = self.norm(x + residual)
        return x


class CNNDecoderLayer(nn.Module):

    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 因果卷积：只看左侧（padding 全加到左边）
        self.causal_pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=2 * d_model,
            kernel_size=kernel_size,
            padding=self.causal_pad
        )
        # 注意力投影
        self.attn_q = nn.Linear(d_model, d_model)
        self.attn_k = nn.Linear(d_model, d_model)
        self.attn_v = nn.Linear(d_model, d_model)
        self.attn_o = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out):
        # ── 1. 因果 Conv + GLU ──
        residual = x
        x = self.dropout(x)
        x_t = x.transpose(1, 2)
        x_t = self.conv(x_t)
        # 去掉右侧多余的 padding（保持序列长度不变）
        x_t = x_t[:, :, :-self.causal_pad] if self.causal_pad > 0 else x_t
        x_t = x_t.transpose(1, 2)
        x_t = F.glu(x_t, dim=-1)
        x = self.norm1(x_t + residual)

        # ── 2. 与编码器的点积注意力 ──
        residual = x
        Q = self.attn_q(x)                                   # (b, tq, d)
        K = self.attn_k(enc_out)                             # (b, tk, d)
        V = self.attn_v(enc_out)                             # (b, tk, d)
        d_k = Q.size(-1)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
        weights = F.softmax(scores, dim=-1)
        ctx = torch.bmm(weights, V)                          # (b, tq, d)
        x = self.norm2(self.attn_o(ctx) + residual)
        return x


# CNN Encoder / Decoder


class CNNEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers=6,
                 kernel_size=3, dropout=0.1, pos_encoding='sinusoidal'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = get_pos_encoding(pos_encoding, d_model, dropout)
        self.layers    = nn.ModuleList(
            [CNNEncoderLayer(d_model, kernel_size, dropout) for _ in range(num_layers)]
        )
        self.scale = math.sqrt(d_model)

    def forward(self, src, src_mask=None):
        x = self.pos_enc(self.embedding(src) * self.scale)
        for layer in self.layers:
            x = layer(x)
        return x


class CNNDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers=6,
                 kernel_size=3, dropout=0.1, pos_encoding='sinusoidal'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = get_pos_encoding(pos_encoding, d_model, dropout)
        self.layers    = nn.ModuleList(
            [CNNDecoderLayer(d_model, kernel_size, dropout) for _ in range(num_layers)]
        )
        self.scale = math.sqrt(d_model)

    def forward(self, tgt, enc_out):
        x = self.pos_enc(self.embedding(tgt) * self.scale)
        for layer in self.layers:
            x = layer(x, enc_out)
        return x


# 完整 CNN Seq2Seq 模型

class CNNSeq2Seq(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=256, num_layers=4,
                 kernel_size=3, dropout=0.1,
                 pos_encoding='sinusoidal'):
        super().__init__()
        self.encoder     = CNNEncoder(src_vocab_size, d_model, num_layers,
                                      kernel_size, dropout, pos_encoding)
        self.decoder     = CNNDecoder(tgt_vocab_size, d_model, num_layers,
                                      kernel_size, dropout, pos_encoding)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, pad_idx=0):
        enc_out = self.encoder(src)
        dec_out = self.decoder(tgt, enc_out)
        return self.output_proj(dec_out)