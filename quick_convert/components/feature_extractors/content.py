# quick_convert/components/feature_extractors/content.py

from __future__ import annotations

import torch

from quick_convert.components.ssl.base import ContentFeatures
from quick_convert.utils import DeviceLike, configure_device

from ...data.base_dataset import AudioBatch
from .base import BaseFeatureExtractor


class ContentFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, encoder: torch.nn.Module, device: DeviceLike = None):
        self.encoder = encoder.eval()
        self.device = configure_device(device)
        self.encoder.to(self.device)

    @property
    def feature_name(self) -> str:
        return "content"

    @torch.inference_mode()
    def extract_batch(self, batch: AudioBatch) -> ContentFeatures:
        return self.encoder(batch)
