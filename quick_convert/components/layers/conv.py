
import torch
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