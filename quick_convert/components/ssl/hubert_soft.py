"""Soft HuBERT features from the converted VoicePrivacy checkpoint."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from quick_convert.data import AudioBatch
from quick_convert.utils.audio import load_audio_input
from quick_convert.utils.sequence_validation import validate_padded_lengths

from .base import ContentEncoder, ContentFeatures


class SoftHubertContentEncoder(ContentEncoder):
    """Extract 200-dimensional soft-unit logits from prepared waveforms.

    Gain normalization and hop-aligned cropping are the calling system's
    frontend policy. Recordings are encoded individually to preserve the
    checkpoint's time-axis normalization.
    """

    SAMPLE_RATE = 16_000
    FEATURE_DIM = 200
    FRAME_HZ = 50

    def __init__(self, checkpoint_path: str | Path, device=None):
        super().__init__(device)
        from torchaudio.models import hubert_base

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.hubert = hubert_base()
        self.hubert.load_state_dict(checkpoint["hubert"])
        self.unit_projection = nn.Parameter(checkpoint["W"].float(), requires_grad=False)
        self.to(self.device).eval().requires_grad_(False)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.device = self.unit_projection.device
        return result

    @property
    def sample_rate(self):
        return self.SAMPLE_RATE

    @property
    def frame_hz(self):
        return self.FRAME_HZ

    def forward(self, batch: AudioBatch) -> ContentFeatures:
        if batch.sample_rates is None or torch.any(batch.sample_rates != self.sample_rate):
            raise ValueError("Soft HuBERT requires loaded 16 kHz audio.")
        return self.encode_waveforms(batch.waveforms, batch.lengths, self.sample_rate)

    def encode_file(self, path):
        audio = load_audio_input(path, target_sample_rate=self.sample_rate, mono=True)
        return self.encode_waveforms(audio, sample_rate=self.sample_rate)

    @torch.inference_mode()
    def encode_waveforms(self, waveforms, lengths=None, sample_rate=None):
        if sample_rate not in (None, self.sample_rate) or waveforms is None or waveforms.ndim != 2:
            raise ValueError("Soft HuBERT requires 16 kHz waveforms with shape [batch, time].")
        if lengths is None:
            lengths = torch.full((len(waveforms),), waveforms.shape[1], dtype=torch.long, device=waveforms.device)
        validate_padded_lengths(waveforms, lengths, name="Soft HuBERT waveform")
        values = []
        # Time-axis normalization is per recording, excluding batch padding.
        for waveform, length in zip(waveforms, lengths, strict=True):
            features, _ = self.hubert(waveform[:length].to(self.device).float().unsqueeze(0))
            values.append(F.linear(F.normalize(features), F.normalize(self.unit_projection))[0])
        return ContentFeatures(
            values=pad_sequence(values, batch_first=True),
            lengths=torch.tensor([len(value) for value in values], device=self.device),
            feature_dim=self.FEATURE_DIM,
            representation_type="continuous",
            temporal_granularity="frame",
            backend="torchaudio",
            model_name="hubert_soft",
            layer=None,
            frame_hz=self.frame_hz,
        )
