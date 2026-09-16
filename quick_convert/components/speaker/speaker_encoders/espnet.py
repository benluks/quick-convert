from __future__ import annotations

import torch

from ....data import AudioBatch
from .base import SpeakerEmbedding, SpeakerEncoder


class ESPnetSpeakerEncoder(SpeakerEncoder):
    FEATURE_DIM = 192

    def __init__(
        self,
        model_tag: str = "espnet/voxcelebs12_ecapa_wavlm_joint",
        device: str | torch.device | None = None,
        sample_rate: int = 16_000,
        **kwargs,
    ) -> None:
        super().__init__(device=device)
        self.model_tag = model_tag

        from espnet2.bin.spk_inference import Speech2Embedding

        self.model = Speech2Embedding.from_pretrained(model_tag=model_tag, device=str(self.device))
        self.sample_rate = int(getattr(self.model.spk_train_args, "sample_rate", sample_rate))

    @torch.inference_mode()
    def encode(self, wav: torch.Tensor, sr: int) -> SpeakerEmbedding:
        if sr != self.sample_rate:
            raise ValueError(f"ESPnet speaker encoder expects {self.sample_rate} Hz audio, got {sr} Hz")
        if wav.ndim == 2 and wav.shape[0] == 1:
            wav = wav.squeeze(0)
        if wav.ndim != 1:
            raise ValueError(f"Expected waveform shape [T] or [1, T], got {tuple(wav.shape)}")
        values = self._single_embedding(self.model(wav.to(self.device)))
        return SpeakerEmbedding(values, int(values.shape[-1]), "espnet", self.model_tag)

    @torch.inference_mode()
    def encode_batch(self, batch: AudioBatch) -> SpeakerEmbedding:
        if batch.waveforms is None or batch.lengths is None or batch.sample_rates is None:
            raise ValueError("Speaker encoding requires loaded audio.")
        if any(sr != self.sample_rate for sr in batch.sample_rates):
            raise ValueError(f"ESPnet speaker encoder expects {self.sample_rate} Hz audio")
        values = self.model.spk_model(
            batch.waveforms.to(self.device),
            speech_lengths=batch.lengths.to(self.device),
            extract_embd=True,
        )
        values = self._batch_embeddings(values, len(batch))
        return SpeakerEmbedding(values, int(values.shape[-1]), "espnet", self.model_tag)

    def forward(self, batch: AudioBatch) -> SpeakerEmbedding:
        return self.encode_batch(batch)
