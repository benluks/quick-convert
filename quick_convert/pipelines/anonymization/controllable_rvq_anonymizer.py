from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

import torch

from quick_convert.components.layers.rvq import ResidualVectorQuantizer
from quick_convert.components.encoders import RVQDisentangler

from .base_anonymizer import BaseAnonymizer

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
    
class ControllableRVQAnonymizer(BaseAnonymizer):
    def __init__(
        self,
        config_file: str | Path,
        sample_rate: int | None = None,
        device: torch.device | None = None,
        feature_providers: List[Any] | None = None,
    ) -> None:
        super().__init__(device=device, feature_providers=feature_providers)
        self.config_file = Path(config_file)

        config_path = (
            self.config_file / "config.json"
            if self.config_file.is_dir()
            else self.config_file.parent / "config.json"
        )

        with open(config_path) as f:
            self.hparams = AttrDict(json.load(f))

        self.rvq_disentangler = self.hparams.rvq_disentangler.to(self.device)

        ckpt_path = (
            next(self.config_file.glob("*.pt"))
            if self.config_file.is_dir()
            else self.config_file
        )

        state = torch.load(ckpt_path, map_location=self.device)
        self.rvq_disentangler.load_state_dict(state["model"])
        self.rvq_disentangler.eval()

        self.sr = self.sample_rate = sample_rate or self.hparams.sample_rate

        self.flow_matching = None  # to be implemented
        self.vocoder = None        # to be implemented