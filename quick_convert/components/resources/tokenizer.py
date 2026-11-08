from collections.abc import Iterable

import torch
from sentencepiece import SentencePieceProcessor

from quick_convert.components.resources.base import BaseResourceExtractor
from quick_convert.data.types import AudioBatch


class TokenizerFeatureExtractor(BaseResourceExtractor):
    def __init__(self, encoder: SentencePieceProcessor, device: str = "cpu"):
        self.encoder = encoder
        self.device = device
        # self.encoder.to(device)

    @property
    def feature_name(self) -> str:
        return "tokenizer"

    @torch.inference_mode()
    def extract_batch(self, batch: AudioBatch) -> list[list[int]]:
        tokens: Iterable[torch.LongTensor] = self.encoder.encode(batch.resources["transcript"])

        return tokens
