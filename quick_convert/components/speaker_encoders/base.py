from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class SpeakerEmbedding:
    values: torch.Tensor
    embedding_dim: int
    backend: str
    model_name: str


class SpeakerEncoder(ABC):
    @abstractmethod
    def encode(self, wav: torch.Tensor, sr: int) -> SpeakerEmbedding:
        ...
