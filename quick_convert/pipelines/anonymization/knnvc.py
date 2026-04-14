from pathlib import Path
from typing import List, Optional, Union

import torch

from .targets.knnvc import KNNVCTarget
from .base_anonymizer import BaseAnonymizer


class KNNVCAnonymizer(BaseAnonymizer[KNNVCTarget]):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True
        )
        self.sample_rate = self.sr = 16000
        self.target = KNNVCTarget

    def _get_matching_set(self, ref_wav_paths: List):
        self.matching_set = self.model.get_matching_set(ref_wav_paths)

    def set_target(
        self,
        target: Union[str, List],
        target_speaker_root: Optional[Path] = None,
        pattern: Optional[str] = None,
    ):
        if pattern is None:
            self._get_matching_set(target)
        else:
            ref_wavs = sorted(
                map(str, (Path(target_speaker_root) / target).glob(pattern))
            )
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
