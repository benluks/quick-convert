import torch
import torch.nn as nn

from typing import Optional
from .ffn import DecoderFeedForward
from .mha import MultiHeadAttention

class TransformerBlock(nn.Module):
    r"""
    A basic Transformer block: LayerNorm → Self-Attention → LayerNorm → FeedForward.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function for the feed-forward.
        attention_bias (`bool`, *optional*, defaults to `False`): Whether to add bias to attention projections.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        dropout: float = 0.0,
        attention_bias: bool = False,
    ):
        super().__init__()

        # 1. Self-Attention
        self.ln1 = nn.RMSNorm(dim)

        # Attention with RoPE positional embeddings
        self.mha = MultiHeadAttention(
            embed_dim=dim,
            num_heads=num_attention_heads,
            bias=attention_bias,
            dropout=dropout,
        )

        # 2. Feed-forward
        self.ln2 = nn.RMSNorm(dim)
        self.ffn = DecoderFeedForward(
            dim, 
            dropout=dropout, 
            activation_fn='snakebeta'
        )

    def forward(
        self,
        x: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:

        # 1. Self-attention with residual connection
        x = x + self.mha(self.ln1(x),
                         padding_mask=attention_mask)
        
        # 2. Feed-forward with residual connection
        x = x + self.ffn(self.ln2(x))

        return x
