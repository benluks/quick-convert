"""Architecture-independent waveform GAN training."""

from dataclasses import dataclass

import torch

from quick_convert.components.transforms.aligned_crop import aligned_crop

from .base import BaseTrainingModule


@dataclass
class WaveformLoss:
    loss: torch.Tensor


class WaveformReconstructionTrainingModule(BaseTrainingModule):
    def __init__(self, system, objective, optimization, crop_frames=22, **kwargs):
        super().__init__(optimization=optimization, **kwargs)
        self.system = system
        self.objective = objective
        self.crop_frames = crop_frames

    def phase_parameters(self):
        return {
            "generator": self.system.decoder.parameters(),
            "discriminator": self.objective.parameters(),
        }

    def _pair(self, batch, random):
        if torch.any(batch.sample_rates != self.system.decoder.sample_rate):
            raise ValueError("Waveform sample rate must match decoder sample rate.")
        content = self.system.get_resource(batch, "content")
        if content.lengths is None:
            raise ValueError("Content requires valid frame lengths.")
        features, lengths, _ = self.system.encode_features(content.values, content.lengths, metadata=content.metadata)
        return aligned_crop(
            features,
            lengths,
            batch.waveforms,
            batch.lengths,
            hop_size=self.system.reconstruction.hop_size,
            frames=self.crop_frames,
            random=random,
        )

    def execute_phases(self, batch):
        return super().execute_phases(self._pair(batch, random=True))

    def phase_step(self, batch, phase):
        features, real = batch
        fake = self.system.decoder(features)
        loss = self.objective(real, fake, phase)
        self.log(f"train/{phase}/loss", loss, on_step=True, on_epoch=True, batch_size=len(real))
        return WaveformLoss(loss)

    def _shared_step(self, batch, stage):
        features, real = self._pair(batch, random=False)
        fake = self.system.decoder(features)
        loss = self.objective(real, fake, "generator")
        self.log(f"{stage}/loss", loss, on_epoch=True, batch_size=len(real))
        return WaveformLoss(loss)
