from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import torch

from .base import BaseAnonymizer
from .targets import KNNVCTarget, PreparedTarget, ReferenceSpeech


class KNNVCAnonymizer(BaseAnonymizer[KNNVCTarget]):
    """kNN-VC anonymizer using reference speech as its matching set.

    Construction downloads and initializes the pretrained kNN-VC model through
    :func:`torch.hub.load`.
    """

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True)
        self.sample_rate = self.sr = 16000
        self.target = KNNVCTarget
        self._target_space = str(uuid4())

    @property
    def target_space(self) -> str:
        return self._target_space

    @torch.inference_mode()
    def prepare_target(self, target: ReferenceSpeech | PreparedTarget) -> PreparedTarget[torch.Tensor]:
        """Prepare a matching set usable by the shared conversion pipeline."""
        if isinstance(target, PreparedTarget):
            matching_set = self._matching_set_for_target(target).detach().clone()
            return replace(target, conditioning=matching_set)
        if not isinstance(target, ReferenceSpeech):
            raise TypeError("Expected ReferenceSpeech or PreparedTarget.")
        if target.allow_missing_streams:
            raise ValueError("allow_missing_streams requires an SSL named-stream model.")
        if target.aggregation is not None:
            raise ValueError("KNN-VC prepares a matching set, not aggregated speaker embeddings.")
        if target.prompt_seconds is not None:
            raise ValueError("KNN-VC does not support decoder reference prompts.")
        matching_set = self.model.get_matching_set([str(path) for path in target.paths]).detach().clone()
        prepared = PreparedTarget(matching_set, self.target_space, "knnvc-matching-set-v1", target.paths)
        self._matching_set_for_target(prepared)
        return prepared

    def _matching_set_for_target(self, target: PreparedTarget) -> torch.Tensor:
        if not isinstance(target, PreparedTarget):
            raise TypeError("Conversion requires a PreparedTarget from prepare_target().")
        if target.space != self.target_space or target.representation != "knnvc-matching-set-v1":
            raise ValueError("Prepared target belongs to an incompatible model or representation.")
        values = target.conditioning
        if not isinstance(values, torch.Tensor) or values.ndim != 2 or min(values.shape) == 0:
            raise ValueError("KNN-VC target conditioning must be a nonempty frame-level matching set.")
        if not values.is_floating_point() or not torch.isfinite(values).all():
            raise ValueError("KNN-VC target conditioning must contain finite floating-point features.")
        return values

    @torch.inference_mode()
    def convert_batch(self, batch, *, target: PreparedTarget):
        """Convert without changing the legacy default target or caller tensors."""
        matching_set = self._matching_set_for_target(target)
        return self._generate_batch(batch, self._convert, matching_set=matching_set)

    def _convert(self, audio, *, matching_set, sample_rate=None):
        waveform = self.load(audio, sample_rate=sample_rate)
        query_seq = self.model.get_features(waveform)
        if query_seq.ndim == 3 and query_seq.shape[0] == 2:
            query_seq = query_seq.mean(0)
        return self.model.match(query_seq, matching_set.to(query_seq.device).clone(), topk=4).unsqueeze(0)

    def _get_matching_set(self, ref_wav_paths: list):
        self.matching_set = self.model.get_matching_set(ref_wav_paths)

    def set_target(
        self,
        target: str | list,
        target_speaker_root: Path | None = None,
        pattern: str | None = None,
    ):
        if pattern is None:
            self._get_matching_set(target)
        else:
            ref_wavs = sorted(map(str, (Path(target_speaker_root) / target).glob(pattern)))
            self._get_matching_set(ref_wavs)

    def resynthesize(self, audio, *, sample_rate=None):
        waveform = self.load(audio, sample_rate=sample_rate)
        query_seq = self.model.get_features(waveform)
        return self.model.vocode(query_seq.to(self.device)).cpu().squeeze()

    def anonymize(self, audio, *, sample_rate=None):
        return self._convert(audio, sample_rate=sample_rate, matching_set=self.matching_set)
