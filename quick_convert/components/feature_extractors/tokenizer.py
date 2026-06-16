from typing import Iterable

import torch

from quick_convert.data.types import AudioBatch
from sentencepiece import SentencePieceProcessor

from .base import BaseFeatureExtractor


class TokenizerFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, encoder: SentencePieceProcessor, device: str = "cpu"):
        self.encoder = encoder
        self.device = device
        # self.encoder.to(device)

    @property
    def feature_name(self) -> str:
        return "speaker_embedding"

    @torch.inference_mode()
    def extract_batch(self, batch: AudioBatch) -> list[dict[str, torch.Tensor]]:
        tokens: Iterable[torch.LongTensor] = self.encoder.encode(batch.resources["transcript"])

        return tokens
