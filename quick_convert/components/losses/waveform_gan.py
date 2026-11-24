"""Compose waveform discriminators and reconstruction losses."""

import torch
import torch.nn.functional as F
from torch import nn

from quick_convert.components.transforms.aligned_crop import aligned_crop

from quick_convert.external.matcha.hifigan.models import discriminator_loss, feature_loss, generator_loss


class WaveformGANObjective(nn.Module):
    def __init__(self, discriminators, mel, mel_weight=45.0, crop_frames=22, hop_size=320):
        super().__init__()
        self.discriminators = nn.ModuleList(discriminators)
        if not self.discriminators:
            raise ValueError("At least one discriminator is required.")
        self.mel = mel
        self.mel_weight = mel_weight
        self.crop_frames = crop_frames
        self.hop_size = hop_size

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

    def phase_parameters(self, system):
        return {
            "generator": system.decoder.parameters(),
            "discriminator": self.parameters(),
        }

    def prepare_batch(self, system, batch, *, training):
        if torch.any(batch.sample_rates != system.decoder.sample_rate):
            raise ValueError("Waveform sample rate must match decoder sample rate.")
        content = system.get_resource(batch, "content")
        if content.lengths is None:
            raise ValueError("Content requires valid frame lengths.")
        features, lengths, _ = system.encode_features(content.values, content.lengths, metadata=content.metadata)
        return aligned_crop(
            features,
            lengths,
            batch.waveforms,
            batch.lengths,
            hop_size=self.hop_size,
            frames=self.crop_frames,
            random=training,
        )

    def phase_loss(self, system, prepared, phase):
        features, real = prepared
        return self(real, system.decoder(features), phase)
