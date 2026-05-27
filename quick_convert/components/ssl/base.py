from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


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
    frame_hz: Optional[float] = None


class ContentEncoder(nn.Module, ABC):
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
