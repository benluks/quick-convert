from __future__ import annotations

import torch

from quick_convert.components.encoders.speaker.base import SpeakerEmbedding
from quick_convert.components.resources.base import BaseResourceExtractor
from quick_convert.data.base_dataset import AudioBatch


class SpeakerEmbeddingExtractor(BaseResourceExtractor):
    def __init__(self, encoder: torch.nn.Module, device: str = "cpu"):
        self.encoder = encoder.eval()
        self.device = device
        self.encoder.to(device)

    @property
    def feature_name(self) -> str:
        return "speaker_embedding"

    @torch.inference_mode()
    def extract_batch(self, batch: AudioBatch) -> list[torch.Tensor]:
        embeddings: SpeakerEmbedding = self.encoder(batch)

        outputs = [val.cpu() for val in embeddings.values]

        return outputs
