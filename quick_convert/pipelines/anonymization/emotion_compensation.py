from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import torch

from quick_convert.data import AudioSample

from .base_anonymizer import BaseAnonymizer
from ...components.donors.emotion_compensation import latentGenerator, AttrDict


@dataclass(frozen=True)
class EmotionCompensationAudioSample(AudioSample):
    # x-vector path
    xv_path: Optional[Path] = None
    f0: Optional[torch.Tensor] = None


class EmotionCompensationAnonymizer(BaseAnonymizer):
    def __init__(
        self,
        checkpoint_file: str | Path,
        device: str | torch.device | None = None,
        sample_rate: int | None = None,
        xvector_dir: str | Path | None = None,
        xvector_step: str | int | None = None,
        remove_weight_norm: bool = True,
    ) -> None:
        super().__init__()
        self.checkpoint_file = Path(checkpoint_file)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.xvector_dir = Path(xvector_dir) if xvector_dir is not None else None
        self.xvector_step = str(xvector_step) if xvector_step is not None else None

        config_path = (
            self.checkpoint_file / "config.json"
            if self.checkpoint_file.is_dir()
            else self.checkpoint_file.parent / "config.json"
        )
        with open(config_path) as f:
            self.h = AttrDict(json.load(f))

        self.model = latentGenerator(self.h).to(self.device)

        ckpt_path = self._resolve_checkpoint_path(self.checkpoint_file)
        state = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state["generator"])
        self.model.eval()

        if remove_weight_norm:
            self.model.remove_weight_norm()

        self.sample_rate = sample_rate or self.h.sampling_rate

        self._target_xvector_path: Path | None = None

    def _resolve_checkpoint_path(self, checkpoint_file: Path) -> Path:
        if checkpoint_file.is_file():
            return checkpoint_file

        matches = sorted(checkpoint_file.glob("g_*"))
        if not matches:
            raise FileNotFoundError(f"No generator checkpoint found in {checkpoint_file}")
        return matches[-1]

    def set_target(self, target: str | Path) -> None:
        self._target_xvector_path = Path(target)

    def _resolve_xvector_path(self, stem: str) -> Path:
        if self._target_xvector_path is not None:
            return self._target_xvector_path

        if self.xvector_dir is None or self.xvector_step is None:
            raise ValueError("No xvector target configured.")
        return self.xvector_dir / f"{stem}_{self.xvector_step}.xvector"

    @torch.inference_mode()
    def anonymize(
        self,
        # this shouldn't be an AudioSample, but something subclassing it that also includes a feature. Maybe a catch-all AudioSample-with-feature dataclass
        # it should include a precomputed feature class.
        sample: EmotionCompensationAudioSample,
    ) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # [1,1,T]
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)  # assume [1,T] -> [1,1,T]
        elif waveform.ndim != 3:
            raise ValueError(f"Unexpected waveform shape: {tuple(waveform.shape)}")

        waveform = waveform.to(self.device)
        y = self.model.gen_vpc(**sample.__dict__)

        if isinstance(y, tuple):
            y = y[0]

        return y.squeeze(0).detach().cpu()
