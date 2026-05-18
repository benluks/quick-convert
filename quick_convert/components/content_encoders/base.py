from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from jaxtyping import Float, Int


@dataclass
class ContentFeatures:
    values: torch.FloatTensor
    lengths: torch.LongTensor
    frame_hz: Optional[float]
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
        wavs: torch.FloatTensor,
        lengths: torch.LongTensor | None = None,
        # input sample rates of wavs
        sample_rates: torch.LongTensor | None = None,
    ) -> ContentFeatures:
        raise NotImplementedError
