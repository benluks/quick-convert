from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

import torch

from quick_convert.components.decoders.psst1_hifigan import PSST1HifiGanDecoder
from quick_convert.components.feature_extractors.yaapt_f0 import YAAPTF0Extractor
from quick_convert.components.speaker.emotion_compensation import EmotionBoundaryCompensator
from quick_convert.components.speaker.speaker_encoders.speechbrain_ecapa import SpeechBrainECAPASpeakerEncoder
from quick_convert.components.speaker.speaker_generators.psst1_flow import PSST1FlowSpeakerGenerator
from quick_convert.components.ssl.hubert_soft import SoftHubertContentEncoder
from quick_convert.data import AudioBatch, GeneratedAudio
from quick_convert.systems.base import InferenceSystem
from quick_convert.utils.audio import load_audio_input
from quick_convert.utils.device import configure_device

from .base import generate_audio_batch
from .targets import GeneratedVoice, PreparedTarget, SourceVoice


class PSST1ConversionSystem(InferenceSystem):
    """PSST1 conversion with generated pseudo-voices and emotion compensation.

    GeneratedVoice seeds waveform-dependent per-utterance sampling. SourceVoice
    resynthesizes using the source x-vector. Emotion labels come from a loaded
    sample resource, the convert call, or the configured recognizer. Components
    load the pinned pretrained release unless checkpoint_dir selects local files.
    Requires the speechbrain and yaapt extras, plus speechbrain-ser for the
    default emotion recognizer. w controls latent mixing and steps controls
    Euler integration in both directions.
    """

    EMOTIONS = EmotionBoundaryCompensator.EMOTIONS
    CHECKPOINT_FILES = {
        "speaker_encoder": "ecapa.ckpt",
        "flow": "flow.pt",
        "emotion_boundaries": "emotion_boundaries.pt",
        "hubert": "hubert_soft.pt",
        "generator": "hifigan.pt",
    }
    PRETRAINED_REPO_ID = "QKTheoNguyen/psst1"
    PRETRAINED_REVISION = "9686dd85ea028db0314fe69e4df1c3703e5ba216"

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        w: float = 0.25,
        steps: int = 10,
        emotion_recognizer=None,
        emotion_label_map: Mapping[str, str] | None = None,
        emotion_resource: str = "emotion",
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self._target_space = str(uuid4())
        device = configure_device(device)

        checkpoint_dir = self.download() if checkpoint_dir is None else Path(checkpoint_dir)
        missing = [name for name in self.CHECKPOINT_FILES.values() if not (checkpoint_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{checkpoint_dir} is missing PSST1 checkpoint files: {missing}")
        files = {key: checkpoint_dir / name for key, name in self.CHECKPOINT_FILES.items()}

        self.speaker_encoder = SpeechBrainECAPASpeakerEncoder(files["speaker_encoder"], device=device)
        self.speaker_generator = PSST1FlowSpeakerGenerator(files["flow"]).to(device)
        self.emotion_compensator = EmotionBoundaryCompensator(files["emotion_boundaries"]).to(device)
        self.content_encoder = SoftHubertContentEncoder(files["hubert"], device=device)
        self.f0_extractor = YAAPTF0Extractor()
        self.decoder = PSST1HifiGanDecoder(files["generator"], device=device)

        self.emotion_recognizer = emotion_recognizer
        self.emotion_label_map = dict(emotion_label_map or {})
        invalid = {label for label in self.emotion_label_map.values() if label not in self.EMOTIONS}
        if invalid:
            raise ValueError(f"emotion_label_map targets {sorted(invalid)} are not in {self.EMOTIONS}.")
        self.emotion_resource = emotion_resource

        if not 0 <= w <= 1:
            raise ValueError("w must be in [0, 1].")
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer.")
        self.w = float(w)
        self.steps = steps

    @classmethod
    def download(
        cls,
        repo_id: str | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> Path:
        """Fetch the checkpoint files from the Hugging Face Hub and return their directory.

        Files are cached (``HF_HOME``), so only the first call downloads.
        Defaults to :attr:`PRETRAINED_REPO_ID` at :attr:`PRETRAINED_REVISION`.
        """
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                repo_id=repo_id or cls.PRETRAINED_REPO_ID,
                revision=revision or cls.PRETRAINED_REVISION,
                allow_patterns=list(cls.CHECKPOINT_FILES.values()),
                cache_dir=cache_dir,
            )
        )

    @property
    def sample_rate(self):
        return PSST1HifiGanDecoder.SAMPLE_RATE

    @property
    def target_space(self):
        return self._target_space

    def prepare_target(self, target):
        if isinstance(target, (GeneratedVoice, SourceVoice)):
            return PreparedTarget(target, self.target_space, "psst1-policy-v1")
        if not isinstance(target, PreparedTarget):
            raise TypeError("PSST1 requires GeneratedVoice, SourceVoice, or PreparedTarget.")
        if target.space != self.target_space or target.representation != "psst1-policy-v1":
            raise ValueError("Prepared target belongs to an incompatible model or representation.")
        if not isinstance(target.conditioning, (GeneratedVoice, SourceVoice)):
            raise ValueError("Invalid PSST1 prepared target policy.")
        return target

    def _resolve_emotion(self, emotion: str | None, waveform: torch.Tensor) -> str:
        source = "argument"
        if emotion is None:
            if self.emotion_recognizer is None:
                raise ValueError(
                    f"PSST1 needs an emotion label, one of {self.EMOTIONS}: pass emotion=, attach a "
                    f"{self.emotion_resource!r} resource, or configure an emotion_recognizer."
                )
            emotion = self.emotion_recognizer.predict(waveform, self.sample_rate)
            source = "emotion_recognizer"

        label = self.emotion_label_map.get(emotion, emotion)
        if label not in self.EMOTIONS:
            raise ValueError(
                f"Emotion {emotion!r} from the {source} is not one of {self.EMOTIONS} and has no entry in "
                "emotion_label_map."
            )
        return label

    @staticmethod
    def _noise_generator(seed: int, waveform: torch.Tensor) -> torch.Generator:
        generator = torch.Generator()
        digest = hashlib.sha256(seed.to_bytes(8, "little", signed=True))
        digest.update(waveform.detach().cpu().contiguous().to(torch.float32).numpy().tobytes())
        generator.manual_seed(int.from_bytes(digest.digest()[:8], "little") & (2**63 - 1))
        return generator

    @torch.inference_mode()
    def convert(self, audio, *, target: PreparedTarget, sample_rate=None, emotion=None):
        """Convert one recording using a prepared generation or resynthesis policy."""
        if not isinstance(target, PreparedTarget):
            raise TypeError("Conversion requires a PreparedTarget from prepare_target().")
        target = self.prepare_target(target)
        waveform = load_audio_input(audio, target_sample_rate=self.sample_rate, sample_rate=sample_rate, mono=True)[0]
        xvector = self.speaker_encoder.encode(waveform, self.sample_rate).values
        if isinstance(target.conditioning, SourceVoice):
            return self._synthesize(waveform, xvector)
        label = self._resolve_emotion(emotion, waveform)
        pseudo = self.speaker_generator.anonymize(
            xvector,
            w=self.w,
            steps=self.steps,
            generator=self._noise_generator(target.conditioning.seed, waveform),
        )
        return self._synthesize(waveform, self.emotion_compensator(pseudo, label))

    def _synthesize(self, waveform, xvector):
        # The research frontend normalizes/crops only content and F0 audio;
        # speaker and emotion extraction consume the original source waveform.
        hop = 320
        usable = waveform.numel() // hop * hop
        if usable == 0:
            raise ValueError(f"PSST1 needs at least {hop} samples.")
        audio = waveform.detach().float().cpu()
        peak = audio.abs().max()
        if peak > 0:
            audio = audio / peak
        audio = (audio * 0.95)[:usable]
        content = self.content_encoder.encode_waveforms(audio.unsqueeze(0), sample_rate=self.sample_rate)
        f0 = self.f0_extractor.extract(audio, self.sample_rate)
        return self.decoder(content.values[0, : content.lengths[0]], f0, xvector, num_samples=waveform.numel())

    @torch.inference_mode()
    def resynthesize(self, audio, *, sample_rate=None):
        return self.convert(audio, target=self.prepare_target(SourceVoice()), sample_rate=sample_rate)

    def _sample_emotion(self, sample) -> dict[str, str]:
        if self.emotion_resource not in sample.resources.keys():
            return {}
        value = sample.resources[self.emotion_resource].value
        if value is None:
            raise ValueError(f"Resource {self.emotion_resource!r} is not loaded for sample {sample.utt_id!r}.")
        return {"emotion": str(value).strip()}

    @torch.inference_mode()
    def convert_batch(self, batch: AudioBatch, *, target: PreparedTarget, emotion=None) -> GeneratedAudio:
        """Convert a batch using loaded emotion resources or the fallback recognizer."""
        target = self.prepare_target(target)
        sample_kwargs = (
            self._sample_emotion if emotion is None and isinstance(target.conditioning, GeneratedVoice) else None
        )
        return generate_audio_batch(
            batch,
            self.convert,
            sample_rate=self.sample_rate,
            target=target,
            sample_kwargs=sample_kwargs,
            emotion=emotion,
        )
