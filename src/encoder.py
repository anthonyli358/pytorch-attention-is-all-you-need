import torch
from torch import nn
from embedder import Embedder
from src.multi_head_attention import MultiHeadAttention
from src.positionwise_feedforward import PositionwiseFeedForward


class Encoder(nn.Module):
    def __init__(
        self, vocab_size, n_layers=6, d_model=512, d_ff=2048, n_heads=8, dropout=0.1
    ):
        super().__init__()
        self.embedding = Embedder(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)]
        )

    def forward(self, tokens, mask=None):
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x, mask)
        return x


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
