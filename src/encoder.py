import torch
from torch import nn
from src.multi_head_attention import MultiHeadAttention
from src.positionwise_feedforward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, n_heads=8, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # attention + (add + norm)
        _x = x
        x = self.attention(x, x, x, mask)
        x = self.dropout1(x)
        x = self.norm1(_x + x)

        # ffwd + (add + norm)
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(_x + x)

        return x

