# THIS CODE IS COPIED FROM https://github.com/jik876/hifi-gan/blob/master/models.py

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from quick_convert.components.decoders.base import BaseDecoder
from quick_convert.utils.device import DeviceLike, configure_device


LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.h = h
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.h = h
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            remove_weight_norm(layer)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.lin_pre = nn.Linear(h.hubert_dim, h.hifi_dim)
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(nn.Conv1d(h.hifi_dim, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        h.upsample_initial_channel // (2**i),
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        """`x` as (bs, seq_len, dim), regular hifi assumes input of shape (bs, n_mels, seq_len)"""
        x = self.lin_pre(x)
        x = x.permute(0, 2, 1)  # (bs, seq_len, dim) --> (bs, dim, seq_len)

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
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


PRETRAINED_URLS = {
    "prematched": "https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt",
    "original": "https://github.com/bshall/knn-vc/releases/download/v0.1/g_02500000.pt",
}


@dataclass
class KnnVCHifiGanConfig:
    """
    based on https://github.com/bshall/knn-vc/blob/master/hifigan/config_v1_wavlm.json,
    made compatible with this codebase
    """

    hubert_dim: int = 1024
    hifi_dim: int = 512

    resblock: str = "1"
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )

    upsample_rates: tuple[int, ...] = (10, 8, 2, 2)
    upsample_kernel_sizes: tuple[int, ...] = (20, 16, 4, 4)
    upsample_initial_channel: int = 512

    sample_rate: int = 16000


PRETRAINED_URLS = {
    "prematched": "https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt",
    "original": "https://github.com/bshall/knn-vc/releases/download/v0.1/g_02500000.pt",
}


class KnnVCHifiGanDecoder(BaseDecoder):
    def __init__(
        self,
        generator: Generator,
        sample_rate: int = 16000,
        device: DeviceLike = None,
    ):
        super().__init__(device=device)
        self.generator = generator.to(self.device)
        self.sample_rate = sample_rate

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.generator(features).squeeze(1)

    @classmethod
    def from_pretrained(
        cls,
        variant: str = "prematched",
        *,
        device: DeviceLike = None,
        progress: bool = True,
    ) -> "KnnVCHifiGanDecoder":
        if variant not in PRETRAINED_URLS:
            raise ValueError(f"Unknown kNN-VC HiFi-GAN variant {variant!r}. Expected one of {list(PRETRAINED_URLS)}.")

        device = configure_device(device)
        config = KnnVCHifiGanConfig()

        checkpoint = torch.hub.load_state_dict_from_url(
            PRETRAINED_URLS[variant],
            map_location=device,
            progress=progress,
        )

        generator = Generator(config)
        generator.load_state_dict(checkpoint["generator"])
        generator.remove_weight_norm()
        generator.eval()

        return cls(generator=generator, sample_rate=config.sample_rate, device=device)
