# =============================================================================
# model.py —— Transformer 核心模块
# 复现论文: "Attention Is All You Need" (Vaswani et al., 2017)
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. 位置编码（三种形式，用于实验一对比）
# ─────────────────────────────────────────────

class NoPosEncoding(nn.Module):
    """无位置编码（对照组）"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(x)


class SinusoidalPosEncoding(nn.Module):
    """正弦/余弦位置编码（原论文方案）"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePosEncoding(nn.Module):
    """可学习位置编码（实验对比用）"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pe(positions)
        return self.dropout(x)


def get_pos_encoding(name, d_model, dropout=0.1):
    """工厂函数：按名字返回位置编码模块"""
    mapping = {
        'none':       NoPosEncoding,
        'sinusoidal': SinusoidalPosEncoding,
        'learnable':  LearnablePosEncoding,
    }
    if name not in mapping:
        raise ValueError(f"未知位置编码类型: {name}，可选: {list(mapping)}")
    return mapping[name](d_model, dropout)


# ─────────────────────────────────────────────
# 2. Scaled Dot-Product Attention
# ─────────────────────────────────────────────

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, heads, seq_q, d_k)
    K: (batch, heads, seq_k, d_k)
    V: (batch, heads, seq_v, d_v)  seq_k == seq_v
    返回: (batch, heads, seq_q, d_v), attention weights
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights


# ─────────────────────────────────────────────
# 3. Multi-Head Attention
# ─────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_k     = d_model // num_heads
        self.h       = num_heads
        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        b, s, _ = x.size()
        x = x.view(b, s, self.h, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        b = query.size(0)
        Q = self.split_heads(self.W_Q(query))
        K = self.split_heads(self.W_K(key))
        V = self.split_heads(self.W_V(value))

        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        # (batch, heads, seq, d_k) → (batch, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        return self.W_O(out)


# ─────────────────────────────────────────────
# 4. Position-wise Feed-Forward Network
# ─────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# 5. 编码器层 & 编码器
# ─────────────────────────────────────────────

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn       = FeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.drop1     = nn.Dropout(dropout)
        self.drop2     = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # 残差 + LayerNorm
        x = self.norm1(x + self.drop1(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_layers, dropout=0.1, pos_encoding='sinusoidal'):
        super().__init__()
        self.embedding   = nn.Embedding(vocab_size, d_model)
        self.pos_enc     = get_pos_encoding(pos_encoding, d_model, dropout)
        self.layers      = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm        = nn.LayerNorm(d_model)
        self.scale       = math.sqrt(d_model)

    def forward(self, src, src_mask=None):
        x = self.pos_enc(self.embedding(src) * self.scale)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# ─────────────────────────────────────────────
# 6. 解码器层 & 解码器
# ─────────────────────────────────────────────

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn        = FeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.drop1      = nn.Dropout(dropout)
        self.drop2      = nn.Dropout(dropout)
        self.drop3      = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.drop1(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop2(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.drop3(self.ffn(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_layers, dropout=0.1, pos_encoding='sinusoidal'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = get_pos_encoding(pos_encoding, d_model, dropout)
        self.layers    = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm      = nn.LayerNorm(d_model)
        self.scale     = math.sqrt(d_model)

    def forward(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.pos_enc(self.embedding(tgt) * self.scale)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)


# ─────────────────────────────────────────────
# 7. 完整 Transformer
# ─────────────────────────────────────────────

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, num_heads=8, d_ff=2048,
                 num_layers=6, dropout=0.1,
                 pos_encoding='sinusoidal'):
        super().__init__()
        self.encoder    = Encoder(src_vocab_size, d_model, num_heads,
                                  d_ff, num_layers, dropout, pos_encoding)
        self.decoder    = Decoder(tgt_vocab_size, d_model, num_heads,
                                  d_ff, num_layers, dropout, pos_encoding)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src, pad_idx=0):
        # (batch, 1, 1, seq_len)
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt, pad_idx=0):
        b, t = tgt.size()
        pad_mask  = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)   # (b,1,1,t)
        seq_mask  = torch.tril(torch.ones(t, t, device=tgt.device)).bool()  # (t,t)
        return pad_mask & seq_mask

    def forward(self, src, tgt, pad_idx=0):
        src_mask = self.make_src_mask(src, pad_idx)
        tgt_mask = self.make_tgt_mask(tgt, pad_idx)
        enc_out  = self.encoder(src, src_mask)
        dec_out  = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.output_proj(dec_out)