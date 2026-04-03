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


class KNNVC(BaseAnonymizer):
    def __init__(self):
        self.model = torch.hub.load(
            "bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True
        )
        self.sr = 16000

    def _get_matching_set(self, ref_wav_paths: List):
        self.matching_set = self.model.get_matching_set(ref_wav_paths)

    def set_target(self, target: Union[str, List], pattern: Optional[str] = None):
        if pattern is None:
            self._get_matching_set(target)
        else:
            ref_wavs = sorted(map(str, Path(target).glob(pattern)))
            self._get_matching_set(ref_wavs)

    def resynthesize(self, audio_path):
        query_seq = self.model.get_features(audio_path)
        return self.model.vocode(query_seq.to(self.device)).cpu().squeeze()

    def convert(self, audio_path):
        query_seq = self.model.get_features(str(audio_path))
        if query_seq.ndim == 3 and query_seq.shape[0] == 2:
            # query is stereo, reduce to mono
            query_seq = query_seq.mean(0)
        return self.model.match(query_seq, self.matching_set, topk=4).unsqueeze(0)
