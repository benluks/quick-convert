from __future__ import annotations

import torch

from quick_convert.components.encoders.content.base import ContentFeatures
from quick_convert.components.resources.base import BaseResourceExtractor
from quick_convert.data.base_dataset import AudioBatch
from quick_convert.data.resources.base import CONTENT_FEATURE_CACHE_FORMAT


class ContentFeatureExtractor(BaseResourceExtractor):
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
