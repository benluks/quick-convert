"""ASR bottleneck conversion with explicit model speaker identities."""

from uuid import uuid4

import torch

from quick_convert.systems.base import InferenceSystem
from quick_convert.utils.audio import load_audio_input
from quick_convert.utils.device import configure_device

from .base import generate_audio_batch
from .targets import PreparedTarget, SpeakerIdentity


class ASRBNConversionSystem(InferenceSystem):
    """Convert speech using the pretrained SA Toolkit bottleneck model."""

    SAMPLE_RATE = 16000

    @property
    def sample_rate(self):
        return self.SAMPLE_RATE

    def __init__(self, device=None):
        super().__init__()
        self.device = configure_device(device)
        self._target_space = str(uuid4())
        self.model = (
            torch.hub.load(
                "deep-privacy/SA-toolkit",
                "anonymization",
                tag_version="hifigan_bn_tdnnf_wav2vec2_vq_48_v1",
                trust_repo=True,
            )
            .to(self.device)
            .eval()
        )

    @property
    def target_space(self):
        return self._target_space

    def prepare_target(self, target: SpeakerIdentity | PreparedTarget) -> PreparedTarget[str]:
        if isinstance(target, SpeakerIdentity):
            return PreparedTarget(
                target.speaker_id, self.target_space, "asrbn-speaker-v1", speaker_id=target.speaker_id
            )
        if not isinstance(target, PreparedTarget):
            raise TypeError("Expected SpeakerIdentity or PreparedTarget.")
        if target.space != self.target_space or target.representation != "asrbn-speaker-v1":
            raise ValueError("Prepared target belongs to an incompatible model or representation.")
        if (
            not isinstance(target.conditioning, str)
            or not target.conditioning.strip()
            or target.speaker_id != target.conditioning
        ):
            raise ValueError("ASR-BN target conditioning must match its explicit speaker identity.")
        return target

    @torch.inference_mode()
    def convert(self, audio, *, target: PreparedTarget, sample_rate=None):
        if not isinstance(target, PreparedTarget):
            raise TypeError("Conversion requires a PreparedTarget from prepare_target().")
        target = self.prepare_target(target)
        waveform = load_audio_input(audio, target_sample_rate=self.sample_rate, sample_rate=sample_rate, mono=True)
        return self.model.convert(waveform.to(self.device), target=target.conditioning).cpu()

    @torch.inference_mode()
    def convert_batch(self, batch, *, target: PreparedTarget):
        target = self.prepare_target(target)
        return generate_audio_batch(batch, self.convert, sample_rate=self.sample_rate, target=target)
