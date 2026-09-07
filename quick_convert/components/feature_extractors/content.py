# quick_convert/components/feature_extractors/content.py

from __future__ import annotations

import torch

from quick_convert.components.ssl.base import ContentFeatures

from ...data.base_dataset import AudioBatch
from .base import BaseFeatureExtractor


class ContentFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, encoder: torch.nn.Module, device: str = "cpu"):
        self.encoder = encoder.eval()
        self.device = torch.device(device)
        self.encoder.to(self.device)

    @property
    def feature_name(self) -> str:
        return "content"

    @torch.inference_mode()
    def extract_batch(self, batch: AudioBatch) -> list[dict[str, torch.Tensor]]:
        features: ContentFeatures = self.encoder(batch)
        outputs = [val[:len].cpu() for val, len in zip(features.values, features.lengths)]

        return outputs
