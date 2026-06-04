import math
import torch
from torch import nn

class Embedder(nn.Module):
    def __init__(self, vocab_size:int, embed_len=512, max_seq_len=3000, dropout=0.1):
        """
        Initialize a random matrix of vocab_size x embed_len.

        Args:
            vocab_size: Size of input vocab dictionary.
            embed_len: Length of vector describing each token. Paper uses 512 for ,emory/detail tradeoff.
            max_seq_len (int, optional): Max length of a sentence. Defaults to 3000.
            dropout (float, optional): Regularisation to prevent overfitting. Defaults to 0.1.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_len)

    def forward(self, tokens:torch.tensor):
        x = self.embedding(tokens) * math.sqrt(self.embed_len)  # section 3.4, increase embedding magnitude
        pass
