from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ContentFeatures:
    values: torch.Tensor
    lengths: torch.Tensor | None
    frame_hz: float | None
    feature_dim: int
    representation_type: str
    temporal_granularity: str
    backend: str
    model_name: str
    layer: int | str | None


class ContentEncoder(ABC):
    @abstractmethod
    def encode_file(self, path: str | Path) -> ContentFeatures:
        raise NotImplementedError

    @abstractmethod
    def encode_waveforms(
        self,
        wavs: torch.Tensor,
        lengths: torch.Tensor | None = None,
        sample_rate: int | None = None,
    ) -> ContentFeatures:
        raise NotImplementedError
