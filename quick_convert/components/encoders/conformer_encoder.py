from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.conformer import ConformerBlock


class ConformerEncoder(nn.Module):
    """
    Stack of ConformerBlocks with an optional input projection.

    Input shape: (batch, time, input_dim)
    Output shape: (batch, time, embed_dim)

    Args:
        input_dim:        Feature dimension of the input. If different from
                          embed_dim, a linear projection is applied first.
        embed_dim:        Internal and output feature dimension.
        num_blocks:       Number of stacked ConformerBlocks.
        num_heads:        Number of attention heads in each block.
        ffn_dim:          Hidden dimension of the feed-forward layers.
        conv_kernel_size: Kernel size of the depthwise convolution.
        dropout:          Dropout probability.
        bias:             Use bias in projections.
        pos_emb_base:     Base frequency for RoPE.
    """

    def __init__(
        self,
        input_dim: int = 256,
        embed_dim: int = 256,
        num_blocks: int = 6,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        bias: bool = True,
        pos_emb_base: float = 10000.0,
        use_flash_attention: bool = True,
        pre_norm: bool = False,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, embed_dim, bias=bias) if input_dim != embed_dim else nn.Identity()

        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                    bias=bias,
                    pos_emb_base=pos_emb_base,
                    use_flash_attention=use_flash_attention,
                    pre_norm=pre_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.embed_dim = embed_dim

    # callable so it works with chatterbox
    def output_size(self):
        return self.embed_dim

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
        return x

class ConformerEncoderSSL(nn.Module):
    """
    Stack of ConformerBlocks with an optional input projection.

    Input shape: (batch, time, input_dim)
    Output shape: (batch, time, embed_dim)

    Args:
        input_dim:        Feature dimension of the input. If different from
                          embed_dim, a linear projection is applied first.
        embed_dim:        Internal and output feature dimension.
        num_blocks:       Number of stacked ConformerBlocks.
        num_heads:        Number of attention heads in each block.
        ffn_dim:          Hidden dimension of the feed-forward layers.
        conv_kernel_size: Kernel size of the depthwise convolution.
        dropout:          Dropout probability.
        bias:             Use bias in projections.
        pos_emb_base:     Base frequency for RoPE.
    """

    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 256,
        embed_dim: int = 256,
        num_blocks: int = 6,
        num_heads: int = 4,
        num_ssl_layers: int = 24,
        ffn_dim: int = 1024,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        bias: bool = True,
        pos_emb_base: float = 10000.0,
        use_flash_attention: bool = True,
        pre_norm: bool = False,
    ):
        super().__init__()

        if input_dim != embed_dim:
            self.input_proj = nn.Linear(input_dim, embed_dim, bias=bias)
        else:
            self.input_proj = nn.Identity()

        self.output_proj = nn.Linear(embed_dim, output_dim)

        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                    bias=bias,
                    pos_emb_base=pos_emb_base,
                    use_flash_attention=use_flash_attention,
                    pre_norm=pre_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.embed_dim = embed_dim
        self.layer_weights = nn.Parameter(torch.randn(1, num_ssl_layers))  # (1, L)

    # callable so it works with chatterbox
    def output_size(self):
        return self.embed_dim

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        layer_weights = F.softmax(self.layer_weights, dim=-1)  # (1, L)
        x = torch.einsum("btlc,kl->btc", x, layer_weights)  # (B, T, H, D_in)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
        x = self.output_proj(x)
        return x
