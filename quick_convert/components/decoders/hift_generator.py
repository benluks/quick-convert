from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
from huggingface_hub import hf_hub_download
from torch import nn

from quick_convert.external.cosyvoice.hifigan.f0_predictor import (
    CausalConvRNNF0Predictor,
)
from quick_convert.external.cosyvoice.hifigan.generator import (
    CausalHiFTGenerator,
)
from quick_convert.utils.device import DeviceLike, configure_device


@dataclass
class F0PredictorConfig:
    num_class: int = 1
    in_channels: int = 80
    cond_channels: int = 512


@dataclass
class CosyVoiceHiFTConfig:
    in_channels: int = 80
    base_channels: int = 512
    nb_harmonics: int = 8
    sampling_rate: int = 24_000
    nsf_alpha: float = 0.1
    nsf_sigma: float = 0.003
    nsf_voiced_threshold: float = 10
    upsample_rates: list[int] = field(default_factory=lambda: [8, 5, 3])
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [16, 11, 7])
    istft_params: dict = field(default_factory=lambda: {"n_fft": 16, "hop_len": 4})
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(default_factory=lambda: [[1, 3, 5]] * 3)
    source_resblock_kernel_sizes: list[int] = field(default_factory=lambda: [7, 7, 11])
    source_resblock_dilation_sizes: list[list[int]] = field(default_factory=lambda: [[1, 3, 5]] * 3)
    lrelu_slope: float = 0.1
    audio_limit: float = 0.99
    conv_pre_look_right: int = 4


class CosyVoiceHiFTDecoder(nn.Module):
    def __init__(
        self,
        config: CosyVoiceHiFTConfig | None = None,
        f0_config: F0PredictorConfig | None = None,
        device: DeviceLike = None,
    ):
        super().__init__()

        self.device = configure_device(device)
        self.config = config or CosyVoiceHiFTConfig()
        self.f0_config = f0_config or F0PredictorConfig()

        f0_predictor = CausalConvRNNF0Predictor(**asdict(self.f0_config))

        self.generator = CausalHiFTGenerator(
            **asdict(self.config),
            f0_predictor=f0_predictor,
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        filename: str = "hift.pt",
        *,
        # sampling_rate: int = 24000,
        device: DeviceLike = None,
    ) -> CosyVoiceHiFTDecoder:

        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )

        checkpoint = torch.load(
            checkpoint_path,
            weights_only=False,
        )

        # CosyVoice's hift.pt is expected to contain the HiFT state dict directly.
        # This little fallback keeps it resilient if the checkpoint is wrapped.
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        model = cls(device=device)

        model.generator.load_state_dict(checkpoint, strict=True)

        model.eval()
        model.requires_grad_(False)

        return model

    @torch.inference_mode()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, 80, T]

        Returns:
            waveform: [B, samples]
        """
        waveform, _ = self.generator.inference(
            mel,
            finalize=True,
        )
        return waveform

    @property
    def samples_per_frame(self) -> int:
        return math.prod(self.config.upsample_rates) * self.config.istft_params["hop_len"]

    @property
    def sample_rate(self) -> int:
        return self.config.sampling_rate
