from __future__ import annotations

from pathlib import Path

import torch

from quick_convert.data import AudioBatch
from quick_convert.utils.device import DeviceLike

from .base import SpeakerEmbedding, SpeakerEncoder


class SpeechBrainECAPASpeakerEncoder(SpeakerEncoder):
    """SpeechBrain ECAPA-TDNN x-vector encoder (VoxCeleb, 192-d).

    Features are 80-band SpeechBrain filterbanks with per-utterance mean
    normalization. ``checkpoint_path`` is a SpeechBrain ``embedding_model.ckpt``
    state dict; the weights are loaded explicitly at construction.

    Requires the ``speechbrain`` extra.
    """

    FEATURE_DIM = 192
    SAMPLE_RATE = 16_000
    sample_rate = SAMPLE_RATE

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: DeviceLike = None,
    ):
        super().__init__(device=device)

        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        from speechbrain.processing.features import InputNormalization

        self.checkpoint_path = Path(checkpoint_path)
        self.fbank = Fbank(n_mels=80)
        self.input_norm = InputNormalization(norm_type="sentence", std_norm=False)
        self.model = ECAPA_TDNN(input_size=80, lin_neurons=self.FEATURE_DIM)
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu", weights_only=True))
        self.model.eval()
        self.to(self.device)

    def _embed(self, waveforms: torch.Tensor, relative_lengths: torch.Tensor) -> torch.Tensor:
        feats = self.fbank(waveforms)
        feats = self.input_norm(feats, relative_lengths)
        return self.model(feats, relative_lengths).squeeze(1)

    @torch.inference_mode()
    def encode(self, wav: torch.Tensor, sr: int) -> SpeakerEmbedding:
        if sr != self.sample_rate:
            raise ValueError(f"ECAPA-TDNN expects {self.sample_rate} Hz audio, got {sr} Hz")
        if wav.ndim == 2 and wav.shape[0] == 1:
            wav = wav.squeeze(0)
        if wav.ndim != 1:
            raise ValueError(f"Expected a mono waveform with shape [T] or [1, T], got {tuple(wav.shape)}")

        wav = wav.to(self.device, torch.float32).unsqueeze(0)
        values = self._single_embedding(self._embed(wav, torch.ones(1, device=self.device)))
        return SpeakerEmbedding(
            values=values, embedding_dim=self.feature_dim, backend="speechbrain", model_name="ecapa"
        )

    @torch.inference_mode()
    def encode_batch(self, samples: AudioBatch) -> SpeakerEmbedding:
        if samples.waveforms is None or samples.lengths is None or samples.sample_rates is None:
            raise ValueError("Speaker encoding requires loaded audio.")
        if any(int(sr) != self.sample_rate for sr in samples.sample_rates):
            raise ValueError(f"ECAPA-TDNN expects {self.sample_rate} Hz audio")

        # Encode one by one: sentence normalization over padding would change
        # each embedding relative to the single-utterance path.
        embeddings = [
            self._embed(
                waveform[: int(length)].to(self.device, torch.float32).unsqueeze(0),
                torch.ones(1, device=self.device),
            )
            for waveform, length in zip(samples.waveforms, samples.lengths, strict=True)
        ]
        values = self._batch_embeddings(torch.cat(embeddings, dim=0), len(samples))
        return SpeakerEmbedding(
            values=values, embedding_dim=self.feature_dim, backend="speechbrain", model_name="ecapa"
        )

    def forward(self, samples: AudioBatch) -> SpeakerEmbedding:
        return self.encode_batch(samples)
