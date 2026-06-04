import math
import torch
from torch import nn

class Embedder(nn.Module):
    def __init__(self, vocab_size:int, d_model=512, max_seq_len=3000, dropout=0.1):
        """
        Initialize a random matrix of vocab_size x embed_len.

        Args:
            vocab_size: Size of input vocab dictionary.
            d_model: Length of vector describing each token. Paper uses 512 for ,emory/detail tradeoff.
            max_seq_len (int, optional): Max length of a sentence. Defaults to 3000.
            dropout (float, optional): Regularisation to prevent overfitting. Defaults to 0.1.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        # Compute positional encoding
        pe = torch.zeros(max_seq_len, d_model)  # shape: (3000, 512)
        pos = torch.arange(0, max_seq_len).unsqueeze(1)  # shape: (3000, 1)
        div = 1 / (10000 ** (torch.arange(0, d_model, 2) / d_model))  # shape: (256,)
        pe[:, 0::2] = torch.sin(pos * div)  # broadcasted multiplication (3000, 1) x (1, 256)
        pe[:, 1::2] = torch.cos(pos * div)

        # Fixed state tensor, add to buffer (self) for gpu
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, 3000, 512)

    def forward(self, tokens:torch.Tensor):
        seq_len_idx = tokens.dim() - 1  # handle non batched tests
        x = self.embedding(tokens) * math.sqrt(self.d_model)  # section 3.4, increase embedding magnitude
        x = x + self.pe[:, :tokens.size(seq_len_idx)]  # slice on dimension (assume 2d) to only include encoding on relevant tokens
        x = self.dropout(x)
        return x
