import torch
from torch import nn

from .activations import Swish

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feedforward network (FFN) module.

    Consists of two linear transformations with a ReLU non-linearity in between.
    The same FFN is applied independently to each position in the input sequence.

    Input shape: (batch, time, embed_dim)

    Args:
        embed_dim:   Input and output feature dimension.
        ffn_dim:     Hidden layer dimension of the FFN.
        dropout:     Dropout probability after the first linear layer.
        bias:        Add bias to the linear layers.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.ln = nn.RMSNorm(embed_dim) # RMSNorm is used instead of LayerNorm
        self.ffn1 = nn.Linear(embed_dim, ffn_dim)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.ffn2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x):
        res = x
        x = self.ln(x)
        x = self.ffn1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.ffn2(x)
        x = self.dropout2(x)
        return res + 0.5 * x
    