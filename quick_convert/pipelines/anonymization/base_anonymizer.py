from __future__ import annotations

# anonymizer should take file as input and output [channel, T] audio
from abc import ABC, abstractmethod
from typing import Any, Generic

import torch
import torch.nn as nn

from quick_convert.types import AudioInput
from quick_convert.utils.audio import load_audio_input

from .targets import T_Target


class BaseAnonymizer(nn.Module, ABC, Generic[T_Target]):
    sr: int
    sample_rate: int

    def __init__(self, device: torch.device | None = None, feature_providers: list[Any] | None = None):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.is_batched = False
        self.feature_providers = list(feature_providers or [])

    # def get_feature_providers(self):
    #     return self.feature_providers

    def load(
        self,
        audio: AudioInput,
        *,
        sample_rate: int | None = None,
        convert_to_mono: bool = True,
    ) -> torch.Tensor:
        """Normalize a file or tensor to this anonymizer's input format."""
        return load_audio_input(
            audio,
            target_sample_rate=self.sample_rate,
            sample_rate=sample_rate,
            mono=convert_to_mono,
        )

    def provide_features(self, sample_or_batch):
        features = dict(getattr(sample_or_batch, "features", {}) or {})
        provider_fn = "provide_batch" if self.is_batched else "provide_sample"

        for provider in self.feature_providers:
            # this shouldn't happen because feature extraction shouldn't exist at any other point in the pipeline
            if provider.key in features:
                continue
            features.update({provider.key: getattr(provider, provider_fn)(sample_or_batch)})

        return features

    @abstractmethod
    def set_target(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def anonymize(
        self,
        audio: AudioInput,
        *,
        sample_rate: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError
