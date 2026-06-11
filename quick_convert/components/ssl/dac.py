from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
import torchaudio

from quick_convert.data.types import AudioBatch

from .base import ContentEncoder, ContentFeatures


class DACContentEncoder(ContentEncoder):
    """Content encoder backed by the Descript Audio Codec (DAC).

    Drop-in replacement for ``W2VBertContentEncoder``. It exposes DAC's
    continuous, pre-quantization encoder latent (DAC's own quantizer and decoder
    are unused) as a ``(B, T, 1, D)`` feature at 50 Hz, ready for the pipeline's
    ``ParallelConformerEncoder`` + RVQ stack. The singleton "layer" axis lets the
    conformer consume it unchanged with ``num_layers=1`` and ``input_dim=D``.

    Use the default constructor for a pretrained checkpoint (frozen) i.e. DACContentEncoder.from_pretrained("16khz"), or
    :meth:`from_scratch` to build DAC's encoder with random weights and train it.

    Args:
        pretrained: DAC ``model_type`` to load (e.g. ``'16khz'``). ``None`` builds
            the architecture from scratch using the ``encoder_*`` args below.
        encoder_dim / encoder_rates / latent_dim: DAC encoder hyperparameters,
            used only when ``pretrained is None``.
        sample_rate: Expected input rate (resample at the dataset level).
        trainable: If ``False`` (default) the encoder is frozen so features can be
            precomputed offline; if ``True`` it stays trainable.
    """

    def __init__(
        self,
        *,
        pretrained: Optional[str] = "16khz",
        encoder_dim: int = 64,
        encoder_rates: Sequence[int] = (2, 4, 5, 8),
        latent_dim: Optional[int] = None,
        sample_rate: int = 16000,
        trainable: bool = False,
        device: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(device=device)

        self.sample_rate = sample_rate
        self.trainable = trainable
        self.model_name = f"dac::{pretrained}" if pretrained else "dac::scratch"

        if pretrained is not None:
            # PRETRAINED path: download Descript's DAC and keep ONLY its (already
            # trained) encoder. DAC's own quantizer + decoder are discarded — our
            # pipeline's RVQ and flow-matching decoder do that job downstream.
            import dac

            model = dac.DAC.load(dac.utils.download(model_type=pretrained))
            self.dac_encoder = model.encoder
            self.hop_length = int(model.hop_length)  # audio samples collapsed into one frame
            self.FEATURE_DIM = int(model.latent_dim)  # D: size of each frame's feature vector
        else:
            # FROM-SCRATCH path: build only DAC's Encoder architecture with RANDOM
            # (untrained) weights — no checkpoint, and no decoder/quantizer overhead.
            # Train it either jointly in the pipeline, or with the standalone
            # train_dac_from_scratch.py script and then load the weights back in.
            from dac.model.dac import Encoder

            if latent_dim is None:
                # DAC's convention: channel count doubles at each stride -> dim * 2**n_strides.
                latent_dim = encoder_dim * (2 ** len(encoder_rates))
            self.dac_encoder = Encoder(encoder_dim, list(encoder_rates), latent_dim)
            self.hop_length = int(math.prod(encoder_rates))  # total downsampling factor
            self.FEATURE_DIM = int(latent_dim)

        self.dac_encoder.to(self.device)
        if not trainable:
            # Freeze: fixed function -> features can be precomputed offline.
            self.dac_encoder.eval().requires_grad_(False)

    @classmethod
    def from_scratch(
        cls,
        *,
        encoder_dim: int = 64,
        encoder_rates: Sequence[int] = (2, 4, 5, 8),
        latent_dim: Optional[int] = None,
        sample_rate: int = 16000,
        trainable: bool = True,
        **kwargs,
    ) -> "DACContentEncoder":
        """Build DAC's encoder with random weights, ready to train from scratch.

        Defaults reproduce the 16 kHz DAC encoder (hop 320 -> 50 Hz, latent 1024).
        """
        return cls(
            pretrained=None,
            encoder_dim=encoder_dim,
            encoder_rates=encoder_rates,
            latent_dim=latent_dim,
            sample_rate=sample_rate,
            trainable=trainable,
            **kwargs,
        )

    def encode_waveforms(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor | None = None,
        sample_rate: int | None = None,
    ) -> ContentFeatures:
        """Encode ``(B, T)`` waveforms into DAC's latent, shape ``(B, T, 1, D)``. B=batch size, T=the length along the time axis, D = feature dimension — the size of the feature vector at each frame."""
        # DAC-16k only works at 16 kHz; we resample once at the dataset level, not here.
        sr = sample_rate or self.sample_rate
        if sr != self.sample_rate:
            raise RuntimeError(
                f"{type(self).__name__} expects {self.sample_rate} Hz audio, got {sr}. "
                "Resample at the dataset level (target_sr=16000, load=True)."
            )

        if waveforms.dim() == 3:  # accept (B, 1, T) too -> (B, T)
            waveforms = waveforms.squeeze(1)
        B, T = waveforms.shape
        if lengths is None:  # assume full-length if not told otherwise
            lengths = torch.full((B,), T, dtype=torch.long, device=waveforms.device)

        # Pad to a whole number of frames (DAC's convention), then add the channel
        # axis the conv encoder expects -> (B, 1, T_pad).
        pad = math.ceil(T / self.hop_length) * self.hop_length - T
        x = F.pad(waveforms.to(self.device), (0, pad)).unsqueeze(1)

        z = self.dac_encoder(x)  # (B, D, T_frames) — channels-first
        # -> (B, T, D), then insert L=1 so ParallelConformer reads it unchanged.
        z = z.transpose(1, 2).unsqueeze(2)  # (B, T_frames, 1, D)

        # Valid frames per item = ceil(samples / hop); clamp guards rounding overshoot.
        frame_lengths = torch.ceil(lengths.float() / self.hop_length).long().clamp_max(z.shape[1])

        return ContentFeatures(
            values=z,
            lengths=frame_lengths.to(z.device),
            feature_dim=self.FEATURE_DIM,
            representation_type="continuous",
            temporal_granularity="frame",
            backend="dac",
            model_name=self.model_name,
            layer=None,
            frame_hz=self.sample_rate / self.hop_length,  # 16000/320 = 50 Hz
        )

    def encode_file(self, path) -> ContentFeatures:
        # Single-file convenience path: load, downmix, resample, then encode.
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:  # to mono
            wav = wav.mean(0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return self.encode_waveforms(wav, torch.tensor([wav.shape[-1]]), self.sample_rate)

    def forward(self, batch: AudioBatch) -> ContentFeatures:
        # Batch entry point used by ContentFeatureExtractor; same guards as the
        # other encoders so they're interchangeable.
        if getattr(batch, "waveforms", None) is None:
            raise RuntimeError(
                f"{type(self).__name__} needs loaded audio. Set `load: true` in the dataset config."
            )
        if not (batch.sample_rates == self.sample_rate).all():
            raise RuntimeError(
                f"Expected {self.sample_rate} Hz audio, got {batch.sample_rates}. "
                "Resample at the dataset level (target_sr=16000, load=True)."
            )
        return self.encode_waveforms(batch.waveforms.to(self.device), batch.lengths.to(self.device))
