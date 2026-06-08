from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.conformer import ConformerBlock
# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class ParallelConformerEncoder(nn.Module):
    """
    Multi-head attention over time with a learned per-head weighted sum over
    SSL layers, optionally with Rotary Position Embeddings (RoPE).

    Input shape: (batch, time, num_layers, embed_dim)

    For each head, a softmax-normalised weight vector combines the SSL layers
    before standard scaled dot-product attention is applied over time.

    Args:
        embed_dim:   Total feature dimension. Must be divisible by num_heads.
        num_heads:   Number of attention heads.
        num_layers:  Number of SSL layers in the input (layer dimension).
        dropout:     Attention dropout probability.
        bias:        Add bias to Q/K/V and output projections.
        use_rope:    Apply Rotary Position Embeddings to Q and K.
        rope_base:   Base frequency for RoPE (default 10000).
    """

    def __init__(
        self,
        input_dim: int = 1024,
        embed_dim: int = 128,
        num_layers: int = 24,
        num_meta_heads: int = 4,
        num_sub_heads: int = 4,
        ffn_dim: int = 512,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        bias: bool = True,
        pos_emb_base: float = 10000.0,
    ):
        super().__init__()

        self.down_proj_in = nn.Linear(input_dim, embed_dim)
        self.down_proj_out = nn.Linear(embed_dim * num_meta_heads, embed_dim)
        self.conformer_encoders = nn.ModuleList(
            [
                ConformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_sub_heads,
                    ffn_dim=ffn_dim,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                    bias=bias,
                    pos_emb_base=pos_emb_base,
                )
                for _ in range(num_meta_heads)
            ]
        )

        self.layer_weights = nn.Parameter(torch.randn(num_meta_heads, num_layers))  # (H, L)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Apply layer weights with softmax over layers
        # x: (B, T, L, D_in), layer_weights: (H, L)
        # -> (B, T, H, D_in) after weighted sum over layers
        layer_weights = F.softmax(self.layer_weights, dim=-1)  # (H, L)
        x = torch.einsum("btlc,hl->bthc", x, layer_weights)  # (B, T, H, D_in)
        x = self.down_proj_in(x)  # (B, T, H, D)

        # Apply conformer block over each stream in parallel
        # Each block processes the corresponding head's stream of features independently
        # The output of each block is (B, T, D), and we stack them back into (B, T, H*D)
        x = torch.cat([block(x[:, :, i, :], padding_mask) for i, block in enumerate(self.conformer_encoders)], dim=-1)

        # Down-project back to D
        x = self.down_proj_out(x)  # (B, T, D)

        return x * padding_mask.unsqueeze(-1)
