from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from quick_convert.utils import ConfigurableDevice, configure_device


@dataclass
class ContentFeatures:
    values: torch.FloatTensor
    lengths: torch.LongTensor
    feature_dim: int
    representation_type: str
    temporal_granularity: str
    backend: str
    model_name: str
    layer: int | str | None
    frame_hz: float | None = None


class ContentEncoder(nn.Module, ABC):
    TIME_D: int = 1
    FEATURE_DIM: int | None = None

    def __init__(self, device: ConfigurableDevice = None):
        super().__init__()
        self.device = torch.device(configure_device(device))

    @property
    def feature_dim(self) -> int:
        return self.FEATURE_DIM

    @abstractmethod
    def encode_file(self, path: str | Path) -> ContentFeatures:
        raise NotImplementedError

    @abstractmethod
    def encode_waveforms(
        self,
        wavs: torch.FloatTensor,
        lengths: torch.LongTensor | None = None,
        # input sample rates of wavs
        sample_rates: torch.LongTensor | None = None,
    ) -> ContentFeatures:
        raise NotImplementedError

    def _pad_or_trim_time(self, x: torch.Tensor, max_length: int, pad_value: int = 0) -> torch.Tensor:
        tdim = self.TIME_D
        T = x.shape[tdim]

        if T > max_length:
            raise RuntimeError(
                f"Setting `max_length` on a subclass of {self.__class__} is meant to extend the features. `max_length` was set to {max_length}, but encountered features with a length={T}"
            )

        if T < max_length:
            pad_shape = list(x.shape)
            pad_shape[tdim] = max_length - T
            pad = (x.new_zeros(pad_shape) + pad_value).to(x)
            return torch.cat((x, pad), dim=tdim)

        return x
