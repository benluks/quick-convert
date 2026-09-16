from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from quick_convert.data import AudioBatch, GeneratedAudio
from quick_convert.types import AudioInput
from quick_convert.utils.audio import load_audio_input

from .targets import T_Target


class BaseAnonymizer(nn.Module, ABC, Generic[T_Target]):
    """Base contract for audio anonymization systems.

    Single-item callers may pass a file path or waveform tensor to
    :meth:`anonymize`. Dataset workflows can adapt this interface to their
    richer sample and batch containers.
    """

    sr: int
    sample_rate: int

    def __init__(self, device: torch.device | None = None, feature_providers: list[Any] | None = None):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.is_batched = False
        self.feature_providers = list(feature_providers or [])

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
            if provider.key in features:
                continue
            features.update({provider.key: getattr(provider, provider_fn)(sample_or_batch)})

        return features

    def _generate_batch(self, batch: AudioBatch, operation, **kwargs) -> GeneratedAudio:
        """Apply a single-item operation and preserve exact output lengths.

        This default adapter is intentionally sequential. Implementations with
        native backend batching may override :meth:`anonymize_batch` while
        preserving the same output contract.
        """
        waveforms = []

        for sample in batch:
            if sample.waveform is None:
                waveform = operation(sample.path, **kwargs)
            else:
                if sample.sample_rate is None:
                    raise ValueError(f"Loaded sample {sample.utt_id!r} has no sample rate.")
                waveform = operation(
                    sample.waveform,
                    sample_rate=int(sample.sample_rate),
                    **kwargs,
                )

            if waveform.ndim == 2 and waveform.shape[0] == 1:
                waveform = waveform.squeeze(0)
            elif waveform.ndim != 1:
                raise ValueError(
                    "Anonymizers must generate mono waveforms with shape "
                    f"(time,) or (1, time); got {tuple(waveform.shape)} for sample {sample.utt_id!r}."
                )
            waveforms.append(waveform)

        if not waveforms:
            raise ValueError("Cannot anonymize an empty batch.")

        lengths = torch.tensor(
            [waveform.shape[-1] for waveform in waveforms],
            dtype=torch.long,
            device=waveforms[0].device,
        )
        return GeneratedAudio(
            waveforms=pad_sequence(waveforms, batch_first=True),
            lengths=lengths,
            sample_rate=self.sample_rate,
        )

    def anonymize_batch(self, batch: AudioBatch, **kwargs) -> GeneratedAudio:
        """Anonymize a batch through the exact sequential fallback."""
        return self._generate_batch(batch, self.anonymize, **kwargs)

    def resynthesize_batch(self, batch: AudioBatch, **kwargs) -> GeneratedAudio:
        """Resynthesize a batch when the concrete system supports it."""
        operation = getattr(self, "resynthesize", None)
        if operation is None:
            raise NotImplementedError(f"{type(self).__name__} does not support resynthesis.")
        return self._generate_batch(batch, operation, **kwargs)

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
