from __future__ import annotations
from typing import Union

import torch

from .base import BaseFeatureExtractor
from ...data.types import AudioBatch, AudioSample


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
        return self.extract_fn(batch.waveforms.to(self.device), **self.extract_kwargs)

    @torch.inference_mode()
    def extract_sample(self, sample: AudioSample) -> list[dict[str, torch.Tensor]]:
        return self.extract_fn(sample.waveform.to(self.device), **self.extract_kwargs)
