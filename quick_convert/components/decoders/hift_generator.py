from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field

import torch
from huggingface_hub import hf_hub_download
from torch import nn

from quick_convert.components.feature_extractors.mel import MelSpectrogramConfig
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
        mel_config: MelSpectrogramConfig | None = None,
    ):
        super().__init__()

        self.device = configure_device(device)
        self.config = config or CosyVoiceHiFTConfig()
        self.f0_config = f0_config or F0PredictorConfig()
        self.mel_config = mel_config
        if mel_config is not None:
            if (mel_config.sample_rate, mel_config.hop_size, mel_config.num_mels) != (
                self.sample_rate,
                self.samples_per_frame,
                self.config.in_channels,
            ):
                raise ValueError("Mel specification does not match the HiFT architecture.")

        f0_predictor = CausalConvRNNF0Predictor(**asdict(self.f0_config))

        self.generator = CausalHiFTGenerator(
            **asdict(self.config),
            f0_predictor=f0_predictor,
        )

    def checkpoint_config(self) -> dict:
        return {
            "config": asdict(self.config),
            "f0_config": asdict(self.f0_config),
            "mel_config": asdict(self.mel_config) if self.mel_config is not None else None,
        }

    @classmethod
    def pretrained_config(
        cls,
        repo_id: str,
        config_filename: str,
        revision: str | None = None,
    ) -> dict:
        """Resolve companion metadata without downloading vocoder weights."""
        from pathlib import Path

        from .cosyvoice_config import read_cosyvoice_config

        path = Path(hf_hub_download(repo_id=repo_id, filename=config_filename, revision=revision))
        config, f0_config, mel_config = read_cosyvoice_config(path)
        # Hub cache snapshots identify the exact commit used for metadata and weights.
        resolved_revision = next(
            (parent.name for parent in path.parents if parent.parent.name == "snapshots"), revision
        )
        return {
            "config": config,
            "f0_config": f0_config,
            "mel_config": asdict(mel_config),
            "revision": resolved_revision,
        }

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        filename: str = "hift.pt",
        *,
        config_filename: str = "cosyvoice3.yaml",
        revision: str | None = None,
        metadata: dict | None = None,
        device: DeviceLike = None,
    ) -> CosyVoiceHiFTDecoder:
        metadata = metadata or cls.pretrained_config(repo_id, config_filename, revision)
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=metadata["revision"],
        )
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        model = cls(
            config=CosyVoiceHiFTConfig(**metadata["config"]),
            f0_config=F0PredictorConfig(**metadata["f0_config"]),
            mel_config=MelSpectrogramConfig(**metadata["mel_config"]),
            device=device,
        )
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
        output_device = mel.device
        use_cpu = output_device.type == "mps"
        if use_cpu:
            if not getattr(self, "_warned_mps_fallback", False):
                warnings.warn(
                    "CosyVoice HiFT requires float64, which MPS does not support; "
                    "vocoding will run on CPU and return audio to MPS.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_mps_fallback = True
            mel = mel.to(device="cpu", dtype=torch.float32)

        # Lazy loading and parent .to(...) calls can leave the vocoder on a
        # different device from its input. Only move this inference component.
        self.generator.to(device=mel.device, dtype=torch.float32 if use_cpu else None)
        try:
            waveform, _ = self.generator.inference(mel, finalize=True)
        finally:
            if use_cpu:
                # The backend converts its pitch predictor to float64 in-place.
                # Restore float32 so a later parent .to("mps") remains valid.
                self.generator.float()
        return waveform.to(output_device)

    @property
    def samples_per_frame(self) -> int:
        return math.prod(self.config.upsample_rates) * self.config.istft_params["hop_len"]

    @property
    def sample_rate(self) -> int:
        return self.config.sampling_rate
