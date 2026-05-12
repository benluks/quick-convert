from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, List, Optional

from quick_convert.data.types import AudioBatch

from ...utils.donor_utils import resolve_donor_paths
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
        sample_rate: int | None = None,
        # xvector_provider: Optional[str | Path] = None,
        # f0_provider: Optional[str | Path] = None,
        remove_weight_norm: bool = True,
        donor_root: Optional[Path] = Path(__file__).parents[2] / "components" / "donors" / "emotion_compensation",
        feature_providers: Optional[List[Any]] = None,
    ) -> None:
        super().__init__(feature_providers=feature_providers)
        self.checkpoint_file = Path(checkpoint_file)

        config_path = (
            self.checkpoint_file / "config.json"
            if self.checkpoint_file.is_dir()
            else self.checkpoint_file.parent / "config.json"
        )
        with open(config_path) as f:
            self.h = AttrDict(json.load(f))

        self.donor_root = donor_root

        self.h = resolve_donor_paths(
            self.h,
            self.donor_root,
            [
                "hubert_model_path",
                "checkpoint_file",
                "config_path",
                "stats_path",
                "soft_model_path",
                "ecapa_fbank_model_path",
            ],
        )
        self.sr = self.h["sampling_rate"]

        self.model = latentGenerator(self.h, self.device).to(self.device)

        ckpt_path = self._resolve_checkpoint_path(self.checkpoint_file)
        state = torch.load(ckpt_path, map_location=self.device)
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
        sample: AudioSample,
    ) -> torch.Tensor:

        waveform = torch.atleast_2d(sample.waveform)
        waveform = waveform.to(self.device)

        # Match original latentDataset full-utterance inference behavior
        ssl_hop_size = 320

        num_samples = waveform.shape[-1]
        trimmed_num_samples = (num_samples // ssl_hop_size) * ssl_hop_size
        waveform = waveform[..., :trimmed_num_samples]

        sample = replace(sample, waveform=waveform)
        features = self.provide_features(sample)

        xv_path = features["speaker_embedding"]
        if not xv_path.exists():
            return

        f0 = features["f0"].to(self.device)

        y = self.model.gen_vpc(xv_path, audio=waveform, f0=f0, **sample.__dict__)

        if isinstance(y, tuple):
            y = y[0]

        return y.squeeze(0).detach().cpu()

    @torch.inference_mode()
    def anonymize_batch(
        self,
        batch: AudioBatch,
    ) -> torch.Tensor:

        waveform = torch.atleast_2d(batch.waveform)
        waveform = waveform.to(self.device)

        # xv_path = sample.features.pop("xvector_path")

        xv_path = batch.features["speaker_embedding"]
        f0 = batch.features["f0"]

        # the code expects audio to be of shape [B 1 T]
        y = self.model.gen_vpc(xv_path, audio=waveform.unsqueeze(-2), f0=f0, **batch.__dict__)

        if isinstance(y, tuple):
            y = y[0]

        return y.squeeze(0).detach().cpu()
