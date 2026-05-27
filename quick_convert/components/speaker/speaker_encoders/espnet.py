# quick_convert/components/speaker_encoders/espnet.py
from __future__ import annotations

from pathlib import Path

import torch
from espnet2.bin.spk_inference import Speech2Embedding

from ....data import AudioBatch
from ....utils.audio import load_audio

from .base import SpeakerEmbedding, SpeakerEncoder


class ESPnetSpeakerEncoder(SpeakerEncoder):
    FEATURE_DIM = 192

    def __init__(
        self,
        model_tag: str = "espnet/voxcelebs12_ecapa_wavlm_joint",
        device: str = None,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__(device=device)
        self.model_tag = model_tag
        self.device = device
        self.model = Speech2Embedding.from_pretrained(
            model_tag=model_tag,
            device=device,
        )
        self.sample_rate = self.model.spk_train_args.sample_rate

    def encode_file(self, path: str | Path) -> SpeakerEmbedding:
        """
        Normally I'd prefer to do all the loading in the data module, but in
        the event that the user needs to pass a simple audio file, I'm exposing
        this method
        """

        waveform, _ = load_audio(path, target_sr=self.sample_rate, convert_to_mono=True)
        return self.encode(waveform)

    def encode(
        self,
        AudioSample: torch.Tensor,
    ) -> SpeakerEmbedding:
        # sample_rate included for interface consistency; this backend call path
        # only needs the waveform itself
        embedding = self.speech2spk_embed(AudioSample.waveform)
        return SpeakerEmbedding(
            values=embedding,
            backend="espnet",
            model_name=self.model_tag,
            embedding_dim=int(embedding.shape[-1]),
        )

    def forward(self, batch: AudioBatch):
        return self.encode_batch(batch)

    @torch.inference_mode()
    def encode_batch(
        self,
        batch: AudioBatch,
    ) -> list[SpeakerEmbedding]:
        """
        Batched inference using the underlying speaker model.

        Expected shapes:
        - waveforms: [B, T] or [B, 1, T]
        - lengths:   [B] with true sample lengths before padding
        """
        embeddings = self.model.spk_model(
            batch.waveforms.to(self.device),
            speech_lengths=batch.lengths.to(self.device),
            extract_embd=True,
        )

        return SpeakerEmbedding(
            values=embeddings,
            backend="espnet",
            model_name=self.model_tag,
            embedding_dim=int(embeddings.shape[-1]),
        )
