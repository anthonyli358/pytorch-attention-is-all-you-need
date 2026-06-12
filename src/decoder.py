import torch
from torch import nn
from src.helpers import create_padding_mask, create_causal_mask
from src.embedder import Embedder
from src.multi_head_attention import MultiHeadAttention
from src.positionwise_feedforward import PositionwiseFeedForward


class Decoder(nn.Module):
    def __init__(
        self, vocab_size, n_layers=6, d_model=512, d_ff=2048, n_heads=8, dropout=0.1
    ):
        super().__init__()
        self.embedding = Embedder(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)]
        )

    def forward(self, tokens, encoder_output, mask=None):
        com_mask = create_padding_mask(tokens) & create_causal_mask(
            tokens.size(1), tokens.device
        )
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x, encoder_output, com_mask, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, n_heads=8, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, com_mask=None, mask=None):
        # masked self attention + (add + norm)
        _x = x
        x = self.attention(x, x, x, com_mask)
        x = self.dropout1(x)
        x = self.norm1(_x + x)
        # cross attention + (add + norm)
        _x = x
        x = self.cross_attention(x, encoder_output, encoder_output, mask)
        x = self.dropout2(x)
        x = self.norm2(_x + x)
        # ffwd + (add + norm)
        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(_x + x)
        return x
