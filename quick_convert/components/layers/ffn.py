import torch
from torch import nn

from .activations import Swish, SnakeBeta

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
        bias: bool = True,
    ):
        super().__init__()

        self.bias = bias
        self.ln = nn.RMSNorm(embed_dim)  # RMSNorm is used instead of LayerNorm
        self.ffn1 = nn.Linear(embed_dim, ffn_dim, bias=bias)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.ffn2 = nn.Linear(ffn_dim, embed_dim, bias=bias)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln(x)
        x = self.ffn1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.ffn2(x)
        x = self.dropout2(x)
        return x


class DecoderFeedForward(nn.Module):
    r"""
    A feed-forward layer: SnakeBeta(dim → 4*dim) → Dropout → Linear(4*dim → dim).

    Parameters:
        dim (`int`): The number of channels in the input and output.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        inner_dim = dim * 4
        self.act = SnakeBeta(dim, inner_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.act(x)
        x = self.dropout(x)
        x = self.proj(x)
        return x