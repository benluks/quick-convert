"""Compose waveform discriminators and reconstruction losses."""

import torch.nn.functional as F
from torch import nn

from quick_convert.external.matcha.hifigan.models import discriminator_loss, feature_loss, generator_loss


class WaveformGANObjective(nn.Module):
    def __init__(self, discriminators, mel, mel_weight=45.0):
        super().__init__()
        self.discriminators = nn.ModuleList(discriminators)
        if not self.discriminators:
            raise ValueError("At least one discriminator is required.")
        self.mel = mel
        self.mel_weight = mel_weight

    def forward(self, real, fake, phase):
        if real.ndim != 2 or real.shape != fake.shape:
            raise ValueError("Expected aligned real and generated waveforms of shape (batch, samples).")
        if phase not in ("generator", "discriminator"):
            raise ValueError(f"Unknown GAN phase: {phase}")
        real = real.detach()
        if phase == "discriminator":
            fake = fake.detach()
        loss = real.new_zeros(())
        for discriminator in self.discriminators:
            real_scores, fake_scores, real_maps, fake_maps = discriminator(real.unsqueeze(1), fake.unsqueeze(1))
            if phase == "generator":
                loss = loss + generator_loss(fake_scores)[0]
                loss = loss + feature_loss(real_maps, fake_maps)
            else:
                loss = loss + discriminator_loss(real_scores, fake_scores)[0]
        if phase == "generator":
            loss = loss + self.mel_weight * F.l1_loss(self.mel(fake), self.mel(real))
        return loss
