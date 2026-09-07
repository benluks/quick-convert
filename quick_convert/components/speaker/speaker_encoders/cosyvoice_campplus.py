from __future__ import annotations

from pathlib import Path

import torch

from quick_convert.data import AudioBatch
from quick_convert.utils.device import DeviceLike

from .base import SpeakerEmbedding, SpeakerEncoder


class CosyVoiceCAMPPlusSpeakerEncoder(SpeakerEncoder):
    FEATURE_DIM = 192

    def __init__(
        self,
        repo_id: str = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        filename: str = "campplus.onnx",
        device: DeviceLike = None,
    ):
        super().__init__(device=device)

        self.repo_id = repo_id
        self.filename = filename

        from huggingface_hub import hf_hub_download

        checkpoint_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
            )
        )

        from quick_convert.external.cosyvoice.utils.onnx import EmbeddingExtractor

        self.model = EmbeddingExtractor(
            model_path=str(checkpoint_path),
        )

    def _encode_waveform(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError(f"Expected waveform shape [T] or [1, T], got {tuple(waveform.shape)}")

        # CosyVoice's ONNX extractor runs on CPU internally.
        embedding = self.model.inference(
            waveform.cpu(),
        )

        return embedding.to(
            device=self.device,
            dtype=torch.float32,
        )

    @torch.inference_mode()
    def encode(
        self,
        wav: torch.Tensor,
        sr: int,
    ) -> SpeakerEmbedding:
        if sr != 16_000:
            raise ValueError(f"CosyVoice CAMPPlus expects 16 kHz audio, got {sr} Hz")

        values = self._encode_waveform(wav)

        return SpeakerEmbedding(
            values=values,
            embedding_dim=self.feature_dim,
            backend="campplus",
            model_name="cosyvoice3",
        )

    @torch.inference_mode()
    def encode_batch(
        self,
        samples: AudioBatch,
    ) -> SpeakerEmbedding:
        if any(sr != 16_000 for sr in samples.sample_rates):
            raise ValueError("CosyVoice CAMPPlus expects 16 kHz audio")

        embeddings = []

        for waveform, length in zip(
            samples.waveforms,
            samples.lengths,
        ):
            waveform = waveform[..., : int(length)]

            embedding = self._encode_waveform(waveform)
            embeddings.append(embedding)

        values = torch.stack(
            embeddings,
            dim=0,
        )

        return SpeakerEmbedding(
            values=values,
            embedding_dim=self.feature_dim,
            backend="campplus",
            model_name="cosyvoice3",
        )

    def forward(
        self,
        samples: AudioBatch,
    ) -> SpeakerEmbedding:
        return self.encode_batch(samples)
