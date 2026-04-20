# components/feature_extractors/speaker_embedding.py

from __future__ import annotations

from typing import Any

import torch

from ...data.base_dataset import AudioBatch

from .base import BaseFeatureExtractor, Features


class SpeakerEmbeddingExtractor(BaseFeatureExtractor):
    def __init__(self, speaker_encoder: torch.nn.Module, device: str = "cpu"):
        self.encoder = speaker_encoder.eval()
        self.device = device
        self.encoder.to(device)

    @property
    def feature_name(self) -> str:
        return "speaker_embedding"

    @torch.inference_mode()
    def extract_batch(self, batch: AudioBatch) -> list[dict[str, torch.Tensor]]:
        outputs: list[dict] = []
        for path in batch.paths:
            embedding = self.speaker_encoder(path)

            outputs.append(
                {
                    "values": embedding.values.cpu(),
                    "embedding_dim": embedding.embedding_dim,
                    "backend": embedding.backend,
                    "model_name": embedding.model_name,
                }
            )

        return outputs
