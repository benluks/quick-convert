from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from ....data.base_dataset import AudioBatch
from ....utils.audio import load_audio_input


@dataclass
class SpeakerEmbedding:
    values: torch.Tensor
    embedding_dim: int
    backend: str
    model_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("Speaker embedding values must be a torch.Tensor.")
        if self.values.ndim not in {1, 2}:
            raise ValueError(f"Speaker embeddings must have shape [D] or [B, D], got {tuple(self.values.shape)}.")
        if self.values.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected embedding dimension {self.embedding_dim}, got {self.values.shape[-1]}.")


class SpeakerEncoder(nn.Module, ABC):
    FEATURE_DIM: int
    sample_rate: int

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

    def encode_file(self, path: str | Path) -> SpeakerEmbedding:
        waveform = load_audio_input(
            path,
            target_sample_rate=self.sample_rate,
            mono=True,
            device=self.device,
        )
        return self.encode(waveform, self.sample_rate)

    def _single_embedding(self, values: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(values, device=self.device, dtype=torch.float32)
        if values.ndim == 2 and values.shape[0] == 1:
            values = values.squeeze(0)
        if values.ndim != 1:
            raise ValueError(f"Expected one speaker embedding with shape [D], got {tuple(values.shape)}.")
        return values

    def _batch_embeddings(self, values: torch.Tensor, batch_size: int) -> torch.Tensor:
        values = torch.as_tensor(values, device=self.device, dtype=torch.float32)
        if values.ndim == 3 and values.shape[1] == 1:
            values = values.squeeze(1)
        if values.ndim != 2 or values.shape[0] != batch_size:
            raise ValueError(f"Expected speaker embeddings with shape [{batch_size}, D], got {tuple(values.shape)}.")
        return values

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
