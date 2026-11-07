from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from quick_convert.components.decoders.hifigan import HifiGanGenerator
from quick_convert.components.feature_extractors.yaapt_f0 import YAAPTF0Extractor
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
    """Speaker-conditioned HiFi-GAN resynthesis used by PSST1.

    The generator is conditioned on soft HuBERT content units (200 classes),
    YAAPT F0, and a 192-d ECAPA x-vector, following the speech-resynthesis
    recipe adapted for the VoicePrivacy baselines. Content and F0 come from the
    input waveform; the x-vector selects the output voice.

    Requires the ``yaapt`` extra.

    Args:
        hubert_path: ``hubert_soft.pt`` holding a torchaudio ``hubert_base``
            state dict (``hubert``) and the soft-unit projection (``W``).
        generator_path: ``hifigan.pt`` holding generator weights with weight
            normalization folded in (``generator``).
    """

    SAMPLE_RATE = 16_000
    HOP_LENGTH = 320
    XVECTOR_DIM = 192
    NUM_UNITS = 200

    def __init__(
        self,
        hubert_path: str | Path,
        generator_path: str | Path,
        device: DeviceLike = None,
    ):
        super().__init__()
        from torchaudio.models import hubert_base

        self.register_buffer("_device_anchor", torch.empty(0, device=configure_device(device)), persistent=False)
        hubert = torch.load(hubert_path, map_location="cpu", weights_only=True)
        self.hubert = hubert_base()
        self.hubert.load_state_dict(hubert["hubert"])
        self.unit_projection = nn.Parameter(hubert["W"].float(), requires_grad=False)
        self.generator = PSST1HifiGanGenerator()
        self.generator.load_state_dict(torch.load(generator_path, map_location="cpu", weights_only=True)["generator"])
        self.f0_extractor = YAAPTF0Extractor()
        self.eval()
        self.to(self.device)

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def device(self):
        return self._device_anchor.device

    def soft_units(self, audio: torch.Tensor) -> torch.Tensor:
        """Soft unit logits ``[1, frames, 200]`` for ``audio`` with shape ``[1, T]``."""
        features, _ = self.hubert(audio)
        # As in the trained model, F.normalize uses its default dim=1, which is
        # the time axis for [B, T, C] features.
        return F.linear(F.normalize(features), F.normalize(self.unit_projection))

    @torch.inference_mode()
    def forward(self, waveform: torch.Tensor, xvector: torch.Tensor) -> torch.Tensor:
        """Resynthesize a 16 kHz mono ``[T]`` waveform with the voice of ``xvector``.

        Returns a peak-normalized ``[T]`` waveform on the CPU.
        """
        if waveform.ndim == 2 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        if waveform.ndim != 1:
            raise ValueError(f"Expected a mono waveform with shape [T] or [1, T], got {tuple(waveform.shape)}")
        if xvector.shape != (self.XVECTOR_DIM,):
            raise ValueError(f"Expected one x-vector with shape [{self.XVECTOR_DIM}], got {tuple(xvector.shape)}")

        num_samples = waveform.shape[-1]
        usable = num_samples // self.HOP_LENGTH * self.HOP_LENGTH
        if usable == 0:
            raise ValueError(f"PSST1 needs at least {self.HOP_LENGTH} samples; got {num_samples}.")

        audio = waveform.detach().float().cpu()
        peak = audio.abs().max()
        if peak > 0:
            audio = audio / peak
        audio = (audio * 0.95)[:usable]

        f0 = self.f0_extractor.extract(audio, self.SAMPLE_RATE).to(self.device).reshape(1, 1, -1)
        audio = audio.to(self.device).unsqueeze(0)

        units = self.soft_units(audio)
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
