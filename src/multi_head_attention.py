import math
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_k = d_model // n_heads  # int division
        self.n_heads = n_heads
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        batch_size = q.size(0)
        seq_len_q, seq_len_k, seq_len_v = q.size(1), k.size(1), v.size(1)

        # Split (batch, seq_len, d_model) into (batch, seq_len, n_heads, d_k)
        q = self.W_q(q).view(batch_size, seq_len_q, self.n_heads, self.d_k)
        k = self.W_k(k).view(batch_size, seq_len_k, self.n_heads, self.d_k)
        v = self.W_v(v).view(batch_size, seq_len_v, self.n_heads, self.d_k)

        # Transpose to have batch dimensions first (batch, n_heads, seq_len, d_k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Scaled dot product attention (F.scaled_dot_product_attention)
        frac = torch.matmul(q / math.sqrt(self.d_k), k.transpose(2, 3))
        if mask is not None:  # Mask so masked positions get -1e9 and softmax returns 0
            frac = frac.masked_fill(mask == 0, -1e9)
        brac = nn.functional.softmax(frac, dim=3)

        # Dropout
        brac = self.dropout(brac)

        # (batch, n_heads, seq_len, d_k)
        attention = torch.matmul(brac, v)
        # (batch, seq_len, n_heads, d_k)
        attention = (
            attention.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_q, self.d_model)
        )
        multi_attn = self.W_o(attention)

        return multi_attn
