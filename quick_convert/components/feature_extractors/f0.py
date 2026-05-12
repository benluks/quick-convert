from __future__ import annotations

import torch

from .base import BaseFeatureExtractor
from ...data.types import AudioBatch


class F0Extractor(BaseFeatureExtractor):
    def __init__(self, extract_fn, extract_kwargs=None, device: str = "cpu"):
        self.extract_fn = extract_fn
        self.extract_kwargs = extract_kwargs or {}
        self.device = device
        # self.encoder.to(device)

    @property
    def feature_name(self) -> str:
        return "f0"

    @torch.inference_mode()
    def extract_batch(self, batch: AudioBatch) -> list[dict[str, torch.Tensor]]:
        return self.extract_fn(batch.audio.to(self.device), **self.extract_kwargs)
