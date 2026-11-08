from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from quick_convert.utils.device import configure_device
from quick_convert.utils.sequence_validation import validate_padded_lengths


if TYPE_CHECKING:
    from quick_convert.data.types import AudioBatch


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
        if self.lengths.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            raise TypeError(f"Content feature lengths must use an integer dtype, got {self.lengths.dtype}.")
        validate_padded_lengths(self.values, self.lengths, name="Content feature")
        if self.feature_dim != self.values.shape[-1]:
            raise ValueError(
                f"feature_dim={self.feature_dim} does not match the values' final dimension ({self.values.shape[-1]})."
            )
        if self.frame_hz is not None and self.frame_hz <= 0:
            raise ValueError(f"frame_hz must be positive when provided, got {self.frame_hz}.")

    def representation_metadata(self) -> dict:
        """Describe this representation independently of batch storage."""
        return {
            field.name: getattr(self, field.name)
            for field in fields(ContentFeatures)
            if field.name not in {"values", "lengths"}
        }


class ContentEncoder(nn.Module, ABC):
    """Base interface for frame-level speech representation encoders.

    Implementations accept files or padded waveforms and return
    :class:`ContentFeatures` with exact valid frame lengths. ``values`` use
    batch dimension 0 and time dimension 1.
    """

    TIME_D: int = 1
    FEATURE_DIM: int | None = None

    def __init__(self, device):
        super().__init__()
        self.device = configure_device(device)

    @property
    def feature_dim(self) -> int:
        return self.FEATURE_DIM

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Required waveform sample rate in samples per second."""
        raise NotImplementedError

    @property
    @abstractmethod
    def frame_hz(self) -> float | None:
        """Output-frame spacing in frames per second, if frame based."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, batch: AudioBatch, **kwargs) -> ContentFeatures:
        """Encode a loaded audio batch."""
        raise NotImplementedError

    @abstractmethod
    def encode_file(self, path: str | Path) -> ContentFeatures:
        raise NotImplementedError

    @abstractmethod
    def encode_waveforms(
        self,
        waveforms: torch.FloatTensor,
        lengths: torch.LongTensor | None = None,
        sample_rate: int | None = None,
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


class DiscreteContentEncoder(ContentEncoder, ABC):
    """Content encoder with an explicit discrete bottleneck.

    Discrete encoders may expose continuous representations before their
    quantizer as well as factorized and packed discrete representations.
    ``representation`` names the bottleneck stage; ``layer`` remains the
    encoder-layer selection mechanism used by other SSL encoders.
    """

    REPRESENTATIONS: tuple[str, ...] = ()

    @property
    def representations(self) -> tuple[str, ...]:
        return self.REPRESENTATIONS

    def validate_representation(self, representation: str) -> None:
        if representation not in self.REPRESENTATIONS:
            choices = ", ".join(self.REPRESENTATIONS)
            raise ValueError(
                f"Unknown representation {representation!r} for {self.__class__.__name__}; expected one of: {choices}."
            )
