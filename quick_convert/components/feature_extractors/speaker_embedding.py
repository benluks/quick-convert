# components/feature_extractors/speaker_embedding.py

from __future__ import annotations

import torch

from quick_convert.components.speaker.speaker_encoders.base import SpeakerEmbedding

from ...data.base_dataset import AudioBatch
from .base import BaseFeatureExtractor


class SpeakerEmbeddingExtractor(BaseFeatureExtractor):
    def __init__(self, encoder: torch.nn.Module, device: str = "cpu"):
        self.encoder = encoder.eval()
        self.device = device
        self.encoder.to(device)

    @property
    def feature_name(self) -> str:
        return "speaker_embedding"

    @torch.inference_mode()
    def extract_batch(self, batch: AudioBatch) -> list[dict[str, torch.Tensor]]:
        embeddings: SpeakerEmbedding = self.encoder(batch)

        outputs = [val.cpu() for val in embeddings.values]

        return outputs
