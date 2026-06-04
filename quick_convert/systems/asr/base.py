from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from ...data.types import AudioBatch, MetadataSample


class ASRSystem(torch.nn.Module, ABC):
    def __init__(self, name: str, sr: int, device: str, pred_key="transcript"):
        super().__init__()
        self.device = device
        self.name = name
        self.sr = sr
        self.pred_key=pred_key


    @abstractmethod
    def transcribe(self, sample: MetadataSample) -> str: ...

    @abstractmethod
    def transcribe_batch(self, batch: AudioBatch) -> dict[str, Any]: ...

    def predict_batch(self, batch: AudioBatch):
        return self.transcribe_batch(batch)
