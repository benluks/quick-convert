# quick_convert/components/speaker_encoders/espnet.py
from __future__ import annotations

from pathlib import Path

import torch
from espnet2.bin.spk_inference import Speech2Embedding

from ...data import AudioBatch
from ...utils.audio import load_audio

from .base import SpeakerEmbedding, SpeakerEncoder


class ESPnetSpeakerEncoder(SpeakerEncoder):
    def __init__(
        self,
        model_tag: str = "espnet/voxcelebs12_ecapa_wavlm_joint",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()
        self.model_tag = model_tag
        self.device = device
        self.speech2spk_embed = Speech2Embedding.from_pretrained(
            model_tag=model_tag,
            device=device,
        )
        self.sample_rate = self.speech2spk_embed.spk_train_args.sample_rate

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

    def encode_batch(
        self,
        samples: AudioBatch,
    ) -> list[SpeakerEmbedding]:
        """
        Batched inference using the underlying speaker model.

        Expected shapes:
        - waveforms: [B, T] or [B, 1, T]
        - lengths:   [B] with true sample lengths before padding
        """
        if waveforms.ndim == 3:
            if waveforms.shape[1] != 1:
                raise ValueError(f"Expected mono batch [B, 1, T], got {tuple(waveforms.shape)}")
            waveforms = waveforms.squeeze(1)

        if waveforms.ndim != 2:
            raise ValueError(f"Expected batched waveforms [B, T] or [B, 1, T], got {tuple(waveforms.shape)}")

        if samples.lengths is None and samples.waveforms.shape[0] > 1:
            raise ValueError("Batch is missing lengths information for multiple samples")

        # Move to target device. The underlying model expects tensors here.
        waveforms = waveforms.to(self.device)

        embeddings = self.speech2spk_embed.spk_model(
            samples.waveforms,
            speech_lengths=samples.lengths,
            extract_embd=True,
        )

        return SpeakerEmbedding(
            values=embeddings,
            backend="espnet",
            model_name=self.model_tag,
            embedding_dim=int(embeddings.shape[-1]),
        )
