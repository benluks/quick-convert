import os
from pathlib import Path
from typing import Generic, List, Optional, TypeVar, Union

import torch
import torch.nn as nn
from ...utils.audio import load_audio

# anonymizer should take file as input and output [channel, T] audio
from abc import ABC, abstractmethod
from .targets import T_Target


class BaseAnonymizer(nn.Module, ABC, Generic[T_Target]):
    sr: int
    sample_rate: int

    def __init__(self, device: torch.device | None = None):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

    def load(self, audio_path, convert_to_mono=True):
        wav = load_audio(audio_path, self.sr)
        # is stereo, but make sure it is in [channel, T] format.
        # for now, I don't see how wav could be batched audio so there's
        # no risk of averaging 2 audios
        if wav.ndim >= 2 and wav.shape[-2] == 2 and convert_to_mono:
            wav = wav.mean(dim=0, keepdim=True)

        return wav

    @abstractmethod
    def set_target(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def anonymize(self, audio_path: Union[torch.Tensor, os.PathLike], **kwargs) -> torch.Tensor:
        raise NotImplementedError
