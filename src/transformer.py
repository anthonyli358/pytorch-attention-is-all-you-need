import torch
from torch import nn

from src.helpers import create_padding_mask
from src.encoder import Encoder
from src.decoder import Decoder


class Transformer(nn.Module):
    """
    Implementation of nn.Transformer from scratch.
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        n_layers=6,
        d_ff=2048,
        n_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size, n_layers, d_model, d_ff, n_heads, dropout
        )
        self.decoder = Decoder(
            tgt_vocab_size, n_layers, d_model, d_ff, n_heads, dropout
        )
        self.output = nn.Linear(d_model, tgt_vocab_size)
        # Paper section 3.4
        self.output.weight = self.decoder.embedding.embedding.weight
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(
                    p
                )  # adjust distribution of 2D+ weights, minimize vanishing and exploding gradients

    def forward(self, src_tokens, tgt_tokens):
        mask = create_padding_mask(src_tokens)
        encoder_output = self.encoder(src_tokens, mask)
        decoder_output = self.decoder(tgt_tokens, encoder_output, mask)
        output = self.output(decoder_output)  # (batch, seq_len, tgt_vocab_size)
        return output
