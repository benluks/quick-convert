from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ...data.types import AudioBatch, MetadataSample


class ASRSystem(torch.nn.Module, ABC):
    """Base interface for systems that transcribe audio samples and batches.

    Args:
        name: Human-readable system name.
        sr: Required audio sample rate in hertz.
        device: Device used for inference.
        pred_key: Resource or result key used for predicted transcripts.
    """

    def __init__(self, name: str, sr: int, device: str, pred_key="transcript"):
        super().__init__()
        self.device = device
        self.name = name
        self.sr = sr
        self.pred_key = pred_key

    @abstractmethod
    def transcribe(self, sample: MetadataSample) -> str:
        """Transcribe one sample."""
        ...

    @abstractmethod
    def transcribe_batch(self, batch: AudioBatch) -> list[str]:
        """Transcribe a batch and preserve input order."""
        ...

    def predict_batch(self, batch: AudioBatch):
        return self.transcribe_batch(batch)

    def get_labels(self, batch: AudioBatch) -> list[str]:
        return self.transcribe_batch(batch)
