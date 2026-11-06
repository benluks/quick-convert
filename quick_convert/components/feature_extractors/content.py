# quick_convert/components/feature_extractors/content.py

from __future__ import annotations

import torch

from quick_convert.components.ssl.base import ContentFeatures

from ...data.base_dataset import AudioBatch
from ...data.resources.base import CONTENT_FEATURE_CACHE_FORMAT
from .base import BaseFeatureExtractor


class ContentFeatureExtractor(BaseFeatureExtractor):
    cache_format = CONTENT_FEATURE_CACHE_FORMAT

    def __init__(self, encoder: torch.nn.Module, device: str = "cpu"):
        self.encoder = encoder.eval()
        self.device = torch.device(device)
        self.encoder.to(self.device)

    @property
    def feature_name(self) -> str:
        return "content"

    @torch.inference_mode()
    def extract_batch(self, batch: AudioBatch) -> list[dict]:
        features: ContentFeatures = self.encoder(batch)
        metadata = features.representation_metadata()
        return [
            {"format": self.cache_format, "values": value[:length].cpu().clone(), "metadata": metadata}
            for value, length in zip(features.values, features.lengths, strict=True)
        ]
