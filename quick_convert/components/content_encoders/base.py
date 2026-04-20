from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from jaxtyping import Float, Int


@dataclass
class ContentFeatures:
    values: Float[torch.Tensor, "b n d"]
    lengths: Int[torch.Tensor, "b"]
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
        wavs: Int[torch.Tensor, "b t"],
        lengths: Int[torch.Tensor, "b"] | None = None,
        # input sample rates of wavs
        sample_rates: Int[torch.Tensor, "b"] | None = None,
    ) -> ContentFeatures:
        raise NotImplementedError
