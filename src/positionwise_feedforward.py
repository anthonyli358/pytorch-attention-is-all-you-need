import torch
from torch import nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        """
        Fully connected feed-forward network. 
        Two linear transformations with a ReLU activation in between.

        Args:
            d_model (int, optional): Model dimensions. Defaults to 512.
            dropout (float, optional): Weight dropout. Defaults to 0.1.
        """
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = torch.dropout(dropout)

    def forward(self, x):
        x = nn.functional.relu(self.W_1(x))
        x = self.dropout(x)
        x = self.W_2(x)
        return x