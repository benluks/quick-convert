# Adapted from the PSST1 research code (adapted_from_facebookresearch/models_test.py),
# itself adapted from https://github.com/facebookresearch/speech-resynthesis by
# Xiaoxiao Miao (NII, Japan). Local changes: weight normalization is folded
# into the exported weights, so layers are plain convolutions, and the
# discriminators and training losses are omitted.

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d


LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d))
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                for _ in dilation
            ]
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            xt = c1(F.leaky_relu(x, LRELU_SLOPE))
            xt = c2(F.leaky_relu(xt, LRELU_SLOPE))
            x = xt + x
        return x


class Generator(nn.Module):
    def __init__(
        self,
        model_in_dim=393,
        upsample_rates=(5, 4, 2, 2, 2),
        upsample_kernel_sizes=(11, 8, 4, 4, 4),
        upsample_initial_channel=512,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(model_in_dim, upsample_initial_channel, 7, 1, padding=3)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True)):
            self.ups.append(
                ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(ResBlock1(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)


def upsample_condition(signal, max_frames):
    """Repeat each conditioning frame to span ``max_frames`` (latentGenerator._upsample)."""
    bsz, channels, cond_length = signal.size()
    signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)
    reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
    if reminder > 0:
        raise NotImplementedError("Padding condition signal - misalignment between condition features.")
    return signal.view(bsz, channels, max_frames)
