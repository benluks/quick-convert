import os
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn

from ...utils.audio import load_audio

# anonymizer should take file as input and output [channel, T] audio
from abc import ABC, abstractmethod


class BaseAnonymizer(nn.Module, ABC):
    sr: int
    sample_rate: int

    def __init__(self, device: torch.device | None = None):
        super().__init__()
        self.device = device or torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

    def load(self, audio_path):
        return load_audio(audio_path, self.sr)

    @abstractmethod
    def set_target(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def convert(
        self, audio_path: Union[torch.Tensor, os.PathLike], **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError


class ASRBN(BaseAnonymizer):
    def __init__(self):

        self.model = torch.hub.load(
            "deep-privacy/SA-toolkit",
            "anonymization",
            tag_version="hifigan_bn_tdnnf_wav2vec2_vq_48_v1",
            trust_repo=True,
        )
        self.sr = 16000

    def convert(self, audio_path, target_speaker=None):
        x = self.load(audio_path)
        return self.model.convert(x, target=target_speaker)

