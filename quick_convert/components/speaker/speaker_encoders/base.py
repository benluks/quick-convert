from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from ....data.base_dataset import AudioBatch


@dataclass
class SpeakerEmbedding:
    values: torch.Tensor
    embedding_dim: int
    backend: str
    model_name: str


class SpeakerEncoder(nn.Module, ABC):
    FEATURE_DIM: int

    def __init__(
        self,
        device: str | torch.device | None = None,
    ):
        super().__init__()

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

    @abstractmethod
    def encode(
        self,
        wav: torch.Tensor,
        sr: int,
    ) -> SpeakerEmbedding: ...

    @abstractmethod
    def encode_batch(
        self,
        samples: AudioBatch,
    ) -> SpeakerEmbedding: ...

    @property
    def feature_dim(self) -> int:
        return self.FEATURE_DIM

    def to(
        self,
        device: str | torch.device,
        *args,
        **kwargs,
    ):
        self.device = torch.device(device)
        return super().to(device, *args, **kwargs)
