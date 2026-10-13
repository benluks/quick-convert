"""Explicit waveform-to-mel contracts shared by decoders and their targets."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchaudio


@dataclass(frozen=True)
class MelSpectrogramConfig:
    """Matcha-compatible log magnitude mel specification (Slaney scale/norm)."""

    sample_rate: int
    n_fft: int
    hop_size: int
    win_size: int
    num_mels: int
    fmin: float
    fmax: float | None
    center: bool = False

    def __post_init__(self):
        if min(self.sample_rate, self.n_fft, self.hop_size, self.win_size, self.num_mels) <= 0:
            raise ValueError("Mel dimensions and sample rate must be positive.")
        if not self.hop_size <= self.n_fft or not self.win_size <= self.n_fft:
            raise ValueError("Mel hop and window must not exceed n_fft.")
        if not 0 <= self.fmin < (self.fmax if self.fmax is not None else self.sample_rate / 2) <= self.sample_rate / 2:
            raise ValueError("Mel frequency limits must lie within the Nyquist range.")
        if self.center:
            raise ValueError("Matcha mel extraction requires center=False.")

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        pad = (self.n_fft - self.hop_size) // 2
        return torch.div(lengths + 2 * pad - self.n_fft, self.hop_size, rounding_mode="floor") + 1

    def extract(self, waveforms: torch.Tensor, lengths: torch.Tensor, sample_rate: int):
        """Resample and extract each valid waveform before padding mel frames."""
        if sample_rate <= 0:
            raise ValueError("Input sample rate must be positive.")
        if waveforms.ndim != 2 or lengths.shape != (waveforms.shape[0],):
            raise ValueError("Expected waveforms (B, T) and lengths (B,).")
        if torch.any(lengths <= 0) or torch.any(lengths > waveforms.shape[1]):
            raise ValueError("Waveform lengths must be positive and within the padded tensor.")
        window = torch.hann_window(self.win_size, device=waveforms.device)
        basis = (
            torchaudio.functional.melscale_fbanks(
                self.n_fft // 2 + 1,
                self.fmin,
                self.fmax if self.fmax is not None else self.sample_rate / 2,
                self.num_mels,
                self.sample_rate,
                norm="slaney",
                mel_scale="slaney",
            )
            .to(waveforms.device)
            .transpose(0, 1)
        )
        mels = []
        pad = (self.n_fft - self.hop_size) // 2
        for waveform, length in zip(waveforms, lengths, strict=True):
            waveform = waveform[: int(length)].float()
            if sample_rate != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
            if waveform.numel() <= pad or waveform.numel() + 2 * pad < self.n_fft:
                raise ValueError("Waveform is too short for the configured reflect-padded mel extraction.")
            waveform = F.pad(waveform[None, None], (pad, pad), mode="reflect").flatten()
            spectrum = torch.stft(
                waveform,
                self.n_fft,
                self.hop_size,
                self.win_size,
                window,
                center=False,
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            magnitude = (spectrum.real.square() + spectrum.imag.square() + 1e-9).sqrt()
            mels.append((basis @ magnitude).clamp_min(1e-5).log())
        mel_lengths = torch.tensor([mel.shape[-1] for mel in mels], device=lengths.device)
        max_length = int(mel_lengths.max())
        return torch.stack([F.pad(mel, (0, max_length - mel.shape[-1])) for mel in mels]), mel_lengths
