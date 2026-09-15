from __future__ import annotations

import torch

from ....data import AudioBatch
from .base import SpeakerEmbedding, SpeakerEncoder


class PyannoteWeSpeakerEncoder(SpeakerEncoder):
    FEATURE_DIM = 256

    def __init__(
        self,
        model_name: str = "pyannote/wespeaker-voxceleb-resnet34-LM",
        window: str = "whole",
        device: str | torch.device | None = None,
    ) -> None:
        if window != "whole":
            raise ValueError("SpeakerEncoder produces utterance embeddings; pyannote window must be 'whole'.")
        super().__init__(device=device)

        from pyannote.audio import Inference, Model

        self.model_name = model_name
        self.model = Model.from_pretrained(model_name)
        self.sample_rate = int(self.model.audio.sample_rate)
        self.FEATURE_DIM = int(getattr(self.model, "dimension", self.FEATURE_DIM))
        self.inference = Inference(self.model, window=window, device=self.device)

    @torch.inference_mode()
    def encode(self, wav: torch.Tensor, sr: int) -> SpeakerEmbedding:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim != 2 or wav.shape[0] != 1:
            raise ValueError(f"Expected waveform shape [T] or [1, T], got {tuple(wav.shape)}")
        values = self.inference({"waveform": wav.cpu(), "sample_rate": sr})
        values = self._single_embedding(torch.as_tensor(values))
        return SpeakerEmbedding(values, int(values.shape[-1]), "pyannote.audio", self.model_name)

    @torch.inference_mode()
    def encode_batch(self, batch: AudioBatch) -> SpeakerEmbedding:
        if batch.waveforms is None or batch.lengths is None or batch.sample_rates is None:
            raise ValueError("Speaker encoding requires loaded audio.")
        values = [
            self.encode(waveform[: int(length)], int(sample_rate)).values
            for waveform, length, sample_rate in zip(batch.waveforms, batch.lengths, batch.sample_rates, strict=True)
        ]
        stacked = self._batch_embeddings(torch.stack(values), len(batch))
        return SpeakerEmbedding(stacked, int(stacked.shape[-1]), "pyannote.audio", self.model_name)

    def forward(self, batch: AudioBatch) -> SpeakerEmbedding:
        return self.encode_batch(batch)
