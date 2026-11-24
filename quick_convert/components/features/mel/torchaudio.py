"""Reflect-padded torchaudio log-magnitude mel extraction."""

import torch.nn.functional as F
from torch import nn
from torchaudio.transforms import MelSpectrogram


class TorchaudioLogMel(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_size=320,
        win_size=1024,
        num_mels=80,
        fmin=0.0,
        fmax=None,
    ):
        super().__init__()
        self.pad = (n_fft - hop_size) // 2
        self.transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_size,
            win_length=win_size,
            n_mels=num_mels,
            f_min=fmin,
            f_max=fmax,
            center=False,
            power=1.0,
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, waveform):
        if waveform.ndim != 2:
            raise ValueError("Expected waveforms of shape (batch, samples).")
        waveform = F.pad(waveform.unsqueeze(1), (self.pad, self.pad), mode="reflect").squeeze(1)
        return self.transform(waveform.float()).clamp_min(1e-5).log()
