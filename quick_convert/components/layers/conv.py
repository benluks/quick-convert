

from typing import Optional

import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.utils import weight_norm

from .activations import Swish

def WNConv1d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv1d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))

class DepthWiseConvolution(nn.Module):
    """
    Depth-wise convolution with kernel size 3, padding 1, and optional bias.
    """

    def __init__(self, 
                 channels: int = 256, 
                 kernel_size: int = 9, 
                 bias: bool = True, 
                 dropout: float = 0.0):
        
        super(DepthWiseConvolution, self).__init__()
        
        # kernel_size should be odd to maintain sequence length with padding
        assert kernel_size % 2 == 1

        # "LayerNorm"
        self.ln = nn.RMSNorm(channels) # Replace LayerNorm with RMSNorm

        # First pointwise convolution to expand channels for GLU
        self.pw_conv1 = nn.Conv1d(channels, 
                                  2*channels, 
                                  kernel_size=1, 
                                  stride=1, 
                                  padding=0, 
                                  bias=bias)
        
        # GLU activation splits the channels into two halves and applies sigmoid gating
        self.glu = nn.GLU(dim=1)

        # Depth-wise convolution
        self.dw_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        
        # BatchNorm after depth-wise convolution
        self.bn = nn.BatchNorm1d(channels)
        
        # Swish activation
        self.activation = Swish()

        # Second pointwise convolution
        self.pw_conv2 = nn.Conv1d(channels,
                                  channels, 
                                  kernel_size=1, 
                                  stride=1, 
                                  padding=0, 
                                  bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = self.ln(x)          # (B, T, C)
        x = x.transpose(1, 2)   # (B, C, T)
        x = self.pw_conv1(x)    # (B, 2C, T)
        x = self.glu(x)         # (B, C, T)
        x = self.dw_conv(x)     # (B, C, T)
        x = self.bn(x)          # (B, C, T)
        x = self.activation(x)  # (B, C, T)
        x = self.pw_conv2(x)    # (B, C, T)
        x = self.dropout(x)     # (B, C, T)
        return x.transpose(1, 2)   # (B, T, C)
    


# ---------------------------------------------------------------------------
# 1D Conv Blocks from Matcha TTS
# ---------------------------------------------------------------------------
# Taken from Chatterbox's MatchaTTS: 
# https://github.com/resemble-ai/chatterbox/blob/master/src/chatterbox/models/s3gen/matcha/decoder.py 

class Conv1DBlock(torch.nn.Module):
    def __init__(
            self,
            dim: int,
            dim_out: int,
            groups: int = 8):
        
        super().__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = self.block(x * mask)
        return output * mask

class ResnetBlock1D(torch.nn.Module):
    def __init__(
            self,
            dim: int,
            dim_out: int,
            time_emb_dim: int,
            groups: int = 8):
        
        super().__init__()
        self.mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out))

        self.block1 = Conv1DBlock(dim, dim_out, groups=groups)
        self.block2 = Conv1DBlock(dim_out, dim_out, groups=groups)

        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)

    def forward(
            self, 
            x: torch.Tensor, 
            mask: torch.Tensor, 
            time_emb: torch.Tensor) -> torch.Tensor:
        
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)

        return output

class Downsample1D(nn.Module):
    def __init__(
            self, 
            dim: int):
        
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(
            self,
            channels: int,
            use_conv: bool = False,
            use_conv_transpose: bool = True,
            out_channels: Optional[int] = None,
            name: str = "conv"):
        
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)

        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs
