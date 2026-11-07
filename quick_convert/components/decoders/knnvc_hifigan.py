"""KNN-VC HiFi-GAN adapter, derived from jik876/hifi-gan.

The residual and synthesis backbone is shared with other HiFi-GAN adapters.
"""

from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d
from torch.nn.utils import remove_weight_norm, weight_norm

from quick_convert.components.decoders.base import BaseDecoder
from quick_convert.components.decoders.hifigan import HifiGanGenerator, HifiGanResidualBlock, get_padding, init_weights
from quick_convert.utils.device import DeviceLike, configure_device


LRELU_SLOPE = 0.1


class ResBlock1(HifiGanResidualBlock):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__(channels, kernel_size, dilation, use_weight_norm=True, initialize=True)
        self.h = h

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


class Generator(HifiGanGenerator):
    def __init__(self, h):
        super().__init__(
            h.hifi_dim,
            h.upsample_rates,
            h.upsample_kernel_sizes,
            h.upsample_initial_channel,
            h.resblock_kernel_sizes,
            h.resblock_dilation_sizes,
            residual_block=partial(ResBlock1 if h.resblock == "1" else ResBlock2, h),
            input_projection=nn.Linear(h.hubert_dim, h.hifi_dim),
            use_weight_norm=True,
            initialize=True,
        )
        self.h = h

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
