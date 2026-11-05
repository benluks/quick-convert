from dataclasses import replace
from uuid import uuid4

import torch

from quick_convert.systems.base import InferenceSystem
from quick_convert.utils.audio import load_audio_input
from quick_convert.utils.device import configure_device

from .base import generate_audio_batch
from .targets import PreparedTarget, ReferenceSpeech, SourceVoice


class KNNVCConversionSystem(InferenceSystem):
    """kNN-VC conversion system using reference speech as its matching set.

    Construction downloads and initializes the pretrained kNN-VC model through
    :func:`torch.hub.load`.
    """

    SAMPLE_RATE = 16000

    @property
    def sample_rate(self):
        return self.SAMPLE_RATE

    def __init__(self, device=None):
        super().__init__()
        self.device = configure_device(device)
        self.model = torch.hub.load(
            "bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=str(self.device)
        )
        self._target_space = str(uuid4())

    @property
    def target_space(self) -> str:
        return self._target_space

    @torch.inference_mode()
    def prepare_target(self, target: ReferenceSpeech | SourceVoice | PreparedTarget) -> PreparedTarget[torch.Tensor]:
        """Prepare a matching set usable by the shared conversion pipeline."""
        if isinstance(target, SourceVoice):
            return PreparedTarget(None, self.target_space, "knnvc-source-v1")
        if isinstance(target, PreparedTarget):
            if target.representation == "knnvc-source-v1":
                self._validate_source(target)
                return target
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

    def _validate_source(self, target):
        if (
            target.space != self.target_space
            or target.representation != "knnvc-source-v1"
            or target.conditioning is not None
        ):
            raise ValueError("Prepared source target belongs to an incompatible model or representation.")

    @torch.inference_mode()
    def convert_batch(self, batch, *, target: PreparedTarget):
        target = self.prepare_target(target)
        return generate_audio_batch(batch, self.convert, sample_rate=self.sample_rate, target=target)

    @torch.inference_mode()
    def convert(self, audio, *, target: PreparedTarget, sample_rate=None):
        if not isinstance(target, PreparedTarget):
            raise TypeError("Conversion requires a PreparedTarget from prepare_target().")
        if target.representation == "knnvc-source-v1":
            self._validate_source(target)
            return self.resynthesize(audio, sample_rate=sample_rate)
        matching_set = self._matching_set_for_target(target)
        waveform = load_audio_input(audio, target_sample_rate=self.sample_rate, sample_rate=sample_rate, mono=True)
        query_seq = self.model.get_features(waveform)
        if query_seq.ndim == 3 and query_seq.shape[0] == 2:
            query_seq = query_seq.mean(0)
        return self.model.match(query_seq, matching_set.to(query_seq.device).clone(), topk=4).unsqueeze(0)

    @torch.inference_mode()
    def resynthesize(self, audio, *, sample_rate=None):
        waveform = load_audio_input(audio, target_sample_rate=self.sample_rate, sample_rate=sample_rate, mono=True)
        query_seq = self.model.get_features(waveform)
        return self.model.vocode(query_seq.unsqueeze(0).to(self.device)).cpu().reshape(-1)
