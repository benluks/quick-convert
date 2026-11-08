"""Shared HiFi-GAN residual and upsampling networks.

Derived from jik876/hifi-gan. KNN-VC retains weight normalization and an input
projection; PSST1 uses folded weights. Adapters retain their checkpoint keys.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm


LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


def init_weights(module, mean=0.0, std=0.01):
    if "Conv" in module.__class__.__name__:
        module.weight.data.normal_(mean, std)


def _normalize(module, enabled):
    return weight_norm(module) if enabled else module


class HifiGanResidualBlock(nn.Module):
    """Two convolutions per dilation, with optional legacy weight normalization."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), *, use_weight_norm=False, initialize=False):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                _normalize(
                    nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=get_padding(kernel_size, d)),
                    use_weight_norm,
                )
                for d in dilation
            ]
        )
        if initialize:
            self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList(
            [
                _normalize(
                    nn.Conv1d(channels, channels, kernel_size, padding=get_padding(kernel_size)), use_weight_norm
                )
                for _ in dilation
            ]
        )
        if initialize:
            self.convs2.apply(init_weights)

    def forward(self, x):
        for first, second in zip(self.convs1, self.convs2, strict=True):
            residual = first(F.leaky_relu(x, LRELU_SLOPE))
            residual = second(F.leaky_relu(residual, LRELU_SLOPE))
            x = x + residual
        return x


class HifiGanGenerator(nn.Module):
    """HiFi-GAN synthesis with checkpoint-compatible module registration."""

    def __init__(
        self,
        input_channels,
        upsample_rates,
        upsample_kernel_sizes,
        upsample_initial_channel,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        *,
        residual_block=HifiGanResidualBlock,
        input_projection=None,
        use_weight_norm=False,
        initialize=False,
    ):
        super().__init__()
        if input_projection is not None:
            self.lin_pre = input_projection
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = _normalize(nn.Conv1d(input_channels, upsample_initial_channel, 7, padding=3), use_weight_norm)
        self.ups = nn.ModuleList()
        for index, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True)):
            self.ups.append(
                _normalize(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // 2**index,
                        upsample_initial_channel // 2 ** (index + 1),
                        kernel,
                        rate,
                        padding=(kernel - rate) // 2,
                    ),
                    use_weight_norm,
                )
            )
        self.resblocks = nn.ModuleList()
        for index in range(len(upsample_rates)):
            channels = upsample_initial_channel // 2 ** (index + 1)
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(residual_block(channels, kernel, dilation))
        self.conv_post = _normalize(nn.Conv1d(channels, 1, 7, padding=3), use_weight_norm)
        if initialize:
            self.ups.apply(init_weights)
            self.conv_post.apply(init_weights)

    def forward(self, x):
        if hasattr(self, "lin_pre"):
            x = self.lin_pre(x).permute(0, 2, 1)
        x = self.conv_pre(x)
        for index, upsample in enumerate(self.ups):
            x = upsample(F.leaky_relu(x, LRELU_SLOPE))
            combined = None
            for kernel in range(self.num_kernels):
                value = self.resblocks[index * self.num_kernels + kernel](x)
                combined = value if combined is None else combined + value
            x = combined / self.num_kernels
        return torch.tanh(self.conv_post(F.leaky_relu(x)))
