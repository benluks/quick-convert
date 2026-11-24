"""Training-only HiFi-GAN waveform objective."""

import torch
import torch.nn.functional as F
from torch import nn
from torchaudio.transforms import MelSpectrogram


class HiFiGANObjective(nn.Module):
    """Least-squares GAN, feature matching and log-mel reconstruction."""

    def __init__(
        self,
        discriminators=None,
        sample_rate=16000,
        n_fft=1024,
        hop_length=320,
        n_mels=80,
        mel_weight=45.0,
    ):
        super().__init__()
        if discriminators is None:
            from quick_convert.external.matcha.hifigan.models import (
                MultiPeriodDiscriminator,
                MultiScaleDiscriminator,
            )

            discriminators = [MultiPeriodDiscriminator(), MultiScaleDiscriminator()]
        self.discriminators = nn.ModuleList(discriminators)
        if not self.discriminators:
            raise ValueError("At least one discriminator is required.")
        self.mel_weight = mel_weight
        self.pad = (n_fft - hop_length) // 2
        self.mel = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=False,
            power=1.0,
            norm="slaney",
            mel_scale="slaney",
        )

    def log_mel(self, waveform):
        waveform = F.pad(waveform, (self.pad, self.pad), mode="reflect")
        return self.mel(waveform.float()).clamp_min(1e-5).log()

    def forward(self, real, fake, phase):
        if real.ndim != 2 or real.shape != fake.shape:
            raise ValueError("Expected aligned real and generated waveforms of shape (batch, samples).")
        if phase not in ("generator", "discriminator"):
            raise ValueError(f"Unknown GAN phase: {phase}")
        real = real.detach().unsqueeze(1)
        fake = fake.unsqueeze(1)
        if phase == "discriminator":
            fake = fake.detach()
        loss = real.new_zeros(())
        for discriminator in self.discriminators:
            real_scores, fake_scores, real_maps, fake_maps = discriminator(real, fake)
            for r, g in zip(real_scores, fake_scores, strict=True):
                if phase == "generator":
                    loss = loss + (1 - g).square().mean()
                else:
                    loss = loss + (1 - r).square().mean() + g.square().mean()
            if phase == "generator":
                for real_layers, fake_layers in zip(real_maps, fake_maps, strict=True):
                    for r, g in zip(real_layers, fake_layers, strict=True):
                        loss = loss + 2 * F.l1_loss(g, r.detach())
        if phase == "generator":
            loss = loss + self.mel_weight * F.l1_loss(self.log_mel(fake), self.log_mel(real))
        return loss
