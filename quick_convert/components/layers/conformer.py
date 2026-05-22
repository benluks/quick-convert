import torch
from torch import nn

from .mha import MultiHeadAttention
from .conv import DepthWiseConvolution
from .ffn import PositionwiseFeedForward

class ConformerBlock(nn.Module):

    def __init__(self, embed_dim: int = 256,
                 num_heads: int = 4, 
                 ffn_dim: int = 1024, 
                 conv_kernel_size: int = 31, 
                 dropout: float = 0.1, 
                 bias: bool = True,
                 pos_emb_base: float = 10000.0) -> nn.Module:
        
        super(ConformerBlock, self).__init__()
        
        # All blocks implemented with RMSNorm instead of LayerNorm (let's see how it works)
        # Implemented with RoPE
        self.mha = MultiHeadAttention(embed_dim=embed_dim, 
                                    num_heads=num_heads, 
                                    bias=bias, 
                                    dropout=dropout,
                                    pos_emb_base=pos_emb_base)
        
        self.conv = DepthWiseConvolution(channels=embed_dim,
                                        kernel_size=conv_kernel_size,
                                        bias=bias,
                                        dropout=dropout)
        
        self.ffn1 = PositionwiseFeedForward(embed_dim=embed_dim,
                                          ffn_dim=ffn_dim,
                                          bias=bias,
                                          dropout=dropout)
        self.ffn2 = PositionwiseFeedForward(embed_dim=embed_dim,
                                          ffn_dim=ffn_dim,
                                          bias=bias,
                                          dropout=dropout)
        self.ln = nn.RMSNorm(embed_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # First FFN
        x = self.ffn1(x)

        # Multi-head self-attention
        x = self.mha(x, padding_mask=padding_mask)

        # Convolution module
        x = self.conv(x)

        # Second FFN
        x = self.ffn2(x)

        # Final layer norm
        x = self.ln(x)

        return x