from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class ContentFeatures:
    """Padded content representations and their valid temporal lengths.

    ``values`` always uses batch as dimension 0 and time as dimension 1.
    Additional dimensions may appear between time and the final feature
    dimension.
    """

    values: torch.FloatTensor
    lengths: torch.LongTensor
    feature_dim: int
    representation_type: str
    temporal_granularity: str
    backend: str
    model_name: str
    layer: int | str | None
    frame_hz: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("Content feature values must be a tensor.")
        if self.values.ndim < 3:
            raise ValueError(
                f"Content feature values must have shape (batch, time, ..., feature), got {tuple(self.values.shape)}."
            )
        if not isinstance(self.lengths, torch.Tensor):
            raise TypeError("Content feature lengths must be a tensor.")
        if self.lengths.ndim != 1:
            raise ValueError(f"Content feature lengths must be one-dimensional, got {tuple(self.lengths.shape)}.")
        if self.lengths.shape[0] != self.values.shape[0]:
            raise ValueError(
                "Content feature lengths must contain one value per batch item: "
                f"got {self.lengths.shape[0]} lengths for batch size {self.values.shape[0]}."
            )
        if self.lengths.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            raise TypeError(f"Content feature lengths must use an integer dtype, got {self.lengths.dtype}.")
        if torch.any(self.lengths < 0):
            raise ValueError("Content feature lengths cannot be negative.")
        if torch.any(self.lengths > self.values.shape[1]):
            raise ValueError(
                "A content feature length exceeds the padded time dimension: "
                f"maximum is {self.values.shape[1]}, got {int(self.lengths.max())}."
            )
        if self.feature_dim != self.values.shape[-1]:
            raise ValueError(
                f"feature_dim={self.feature_dim} does not match the values' final dimension ({self.values.shape[-1]})."
            )
        if self.frame_hz is not None and self.frame_hz <= 0:
            raise ValueError(f"frame_hz must be positive when provided, got {self.frame_hz}.")


class ContentEncoder(nn.Module, ABC):
    TIME_D: int = 1
    FEATURE_DIM: int | None = None

    def __init__(self, device):
        super().__init__()
        self.device = torch.device(device)

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
