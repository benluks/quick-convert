from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from quick_convert.components.generators.waveform.hifigan import HifiGanGenerator
from quick_convert.utils.device import DeviceLike, configure_device


class PSST1HifiGanGenerator(HifiGanGenerator):
    """PSST1's folded HiFi-GAN weights, derived from speech-resynthesis.

    The research adaptation is distributed under CC BY-NC 4.0. Source lineage
    and weight provenance are documented in docs/components/psst1.md.
    """

    def __init__(
        self,
        model_in_dim=393,
        upsample_rates=(5, 4, 2, 2, 2),
        upsample_kernel_sizes=(11, 8, 4, 4, 4),
        upsample_initial_channel=512,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__(
            model_in_dim,
            upsample_rates,
            upsample_kernel_sizes,
            upsample_initial_channel,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
        )


def upsample_condition(signal, max_frames):
    """Repeat each conditioning frame to span ``max_frames`` (latentGenerator._upsample)."""
    bsz, channels, cond_length = signal.size()
    signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)
    reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
    if reminder > 0:
        raise NotImplementedError("Padding condition signal - misalignment between condition features.")
    return signal.view(bsz, channels, max_frames)


class PSST1HifiGanDecoder(nn.Module):
    """Synthesize soft HuBERT logits, F0 and an ECAPA speaker vector."""

    SAMPLE_RATE = 16_000
    XVECTOR_DIM = 192
    NUM_UNITS = 200

    def __init__(
        self,
        generator_path: str | Path,
        device: DeviceLike = None,
    ):
        super().__init__()

        self.register_buffer("_device_anchor", torch.empty(0, device=configure_device(device)), persistent=False)
        self.generator = PSST1HifiGanGenerator()
        self.generator.load_state_dict(torch.load(generator_path, map_location="cpu", weights_only=True)["generator"])
        self.eval()
        self.to(self.device)

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def device(self):
        return self._device_anchor.device

    @torch.inference_mode()
    def forward(self, units: torch.Tensor, f0: torch.Tensor, xvector: torch.Tensor, *, num_samples=None):
        """Decode one recording's [frames, 200] logits, [frames] F0 and [192] vector."""
        if units.ndim != 2 or units.shape[-1] != self.NUM_UNITS:
            raise ValueError("Soft HuBERT units must have shape [frames, 200].")
        if f0.ndim != 1 or xvector.shape != (self.XVECTOR_DIM,):
            raise ValueError("Expected F0 with shape [frames] and one [192] speaker vector.")
        units = units.to(self.device).unsqueeze(0)
        f0 = f0.to(self.device).reshape(1, 1, -1)
        units = F.layer_norm(units, units.shape).transpose(2, 1)
        units = F.pad(units, (0, 1), "replicate")

        xv = xvector.to(self.device, torch.float32).reshape(1, self.XVECTOR_DIM, 1)
        xv = F.layer_norm(xv, xv.shape)

        if units.shape[-1] < f0.shape[-1]:
            units = upsample_condition(units, f0.shape[-1])
        else:
            f0 = upsample_condition(f0, units.shape[-1])
        features = torch.cat([units, f0], dim=1)
        features = torch.cat([features, upsample_condition(xv, features.shape[-1])], dim=1)

        output = self.generator(features).reshape(-1)[:num_samples].cpu()
        output_peak = output.abs().max()
        return output / output_peak if output_peak > 0 else output
