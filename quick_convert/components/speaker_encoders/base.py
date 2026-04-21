from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from ...data.base_dataset import AudioBatch


@dataclass
class SpeakerEmbedding:
    values: torch.Tensor
    embedding_dim: int
    backend: str
    model_name: str


class SpeakerEncoder(ABC):
    @abstractmethod
    def encode(self, wav: torch.Tensor, sr: int) -> SpeakerEmbedding: ...

    @abstractmethod
    def encode_batch(self, samples: AudioBatch) -> SpeakerEmbedding: ...

    def to(self, device: str | torch.device):
        self.device = str(device)
        # Speech2Embedding itself manages device internally, but keep this for interface consistency
        return self
