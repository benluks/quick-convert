from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch

from quick_convert.components.decoders.psst1_vocoder import PSST1Vocoder
from quick_convert.components.speaker.emotion_compensation import EmotionBoundaryCompensator
from quick_convert.components.speaker.speaker_encoders.speechbrain_ecapa import SpeechBrainECAPASpeakerEncoder
from quick_convert.components.speaker.speaker_generators.psst1_flow import PSST1FlowSpeakerGenerator
from quick_convert.data import AudioBatch, GeneratedAudio

from .base import BaseAnonymizer


class PSST1Anonymizer(BaseAnonymizer[None]):
    """PSST1: flow-matching pseudo-speakers with emotion compensation.

    Each utterance gets its own pseudo-speaker. The ECAPA x-vector of the input
    is mapped to the flow prior, blended with Gaussian noise (``w`` is the
    weight of the original speaker), mapped back, shifted to a fixed distance
    from the boundary of the utterance's emotion, and used to condition a
    HiFi-GAN that resynthesizes the input's content and F0. See "PSST1 System
    description for VoicePrivacy Challenge 2026" (Nguyen et al.):
    https://www.voiceprivacychallenge.org/vp2026/docs/T02_system_description.pdf

    The emotion label is taken, in order, from the ``emotion`` argument, the
    sample resource named ``emotion_resource`` (dataset runs), or
    ``emotion_recognizer``. Without any of these, anonymization raises.
    ``emotion_label_map`` translates other label sets to :attr:`EMOTIONS`; a
    label that is neither mapped nor already one of :attr:`EMOTIONS` raises.

    Noise is drawn from a CPU generator seeded from ``seed`` and the input
    waveform, so an input and seed always give the same output, different
    inputs get different pseudo-speakers, and the global RNG is untouched.
    ``seed=None`` draws fresh noise on every call.

    Requires the ``speechbrain`` and ``yaapt`` extras.

    Args:
        checkpoint_dir: Directory with the PSST1 weights
            (:attr:`CHECKPOINT_FILES`). ``None`` downloads the released
            weights on first use (see :meth:`download`).
        w: Default weight of the original speaker in the latent blend.
        steps: Default number of Euler steps for each flow direction.
        seed: Default seed; ``None`` for non-reproducible noise.
        emotion_recognizer: Optional fallback with a
            ``predict(waveform, sample_rate) -> str`` method.
        emotion_label_map: Optional mapping from external labels to
            :attr:`EMOTIONS`.
        emotion_resource: Sample resource holding per-utterance labels.
        device: Inference device; defaults to CUDA, then MPS, then CPU.
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
        seed: int | None = None,
        emotion_recognizer=None,
        emotion_label_map: Mapping[str, str] | None = None,
        emotion_resource: str = "emotion",
        device: torch.device | str | None = None,
    ):
        super().__init__(device=torch.device(device) if device is not None else None)
        self.sample_rate = self.sr = PSST1Vocoder.SAMPLE_RATE

        checkpoint_dir = self.download() if checkpoint_dir is None else Path(checkpoint_dir)
        missing = [name for name in self.CHECKPOINT_FILES.values() if not (checkpoint_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{checkpoint_dir} is missing PSST1 checkpoint files: {missing}")
        files = {key: checkpoint_dir / name for key, name in self.CHECKPOINT_FILES.items()}

        self.speaker_encoder = SpeechBrainECAPASpeakerEncoder(files["speaker_encoder"], device=self.device)
        self.speaker_generator = PSST1FlowSpeakerGenerator(files["flow"]).to(self.device)
        self.emotion_compensator = EmotionBoundaryCompensator(files["emotion_boundaries"]).to(self.device)
        self.vocoder = PSST1Vocoder(files["hubert"], files["generator"], device=self.device)

        self.emotion_recognizer = emotion_recognizer
        self.emotion_label_map = dict(emotion_label_map or {})
        invalid = {label for label in self.emotion_label_map.values() if label not in self.EMOTIONS}
        if invalid:
            raise ValueError(f"emotion_label_map targets {sorted(invalid)} are not in {self.EMOTIONS}.")
        self.emotion_resource = emotion_resource

        self.w = self.steps = None
        self.set_target(w=w, steps=steps)
        self.seed = seed

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

    def set_target(self, w: float | None = None, steps: int | None = None):
        """Update the default blend weight and number of flow steps."""
        if w is not None:
            if not 0.0 <= w <= 1.0:
                raise ValueError(f"w must be in [0, 1], got {w}.")
            self.w = float(w)
        if steps is not None:
            if steps < 1:
                raise ValueError(f"steps must be positive, got {steps}.")
            self.steps = int(steps)

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
    def _noise_generator(seed: int | None, waveform: torch.Tensor) -> torch.Generator:
        generator = torch.Generator()
        if seed is None:
            generator.seed()
            return generator
        digest = hashlib.sha256(seed.to_bytes(8, "little", signed=True))
        digest.update(waveform.detach().cpu().contiguous().to(torch.float32).numpy().tobytes())
        generator.manual_seed(int.from_bytes(digest.digest()[:8], "little") & (2**63 - 1))
        return generator

    @torch.inference_mode()
    def anonymize(
        self,
        audio,
        *,
        sample_rate: int | None = None,
        w: float | None = None,
        steps: int | None = None,
        emotion: str | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Anonymize one utterance; returns a 16 kHz ``[T]`` waveform."""
        w = self.w if w is None else w
        steps = self.steps if steps is None else steps
        seed = self.seed if seed is None else seed

        waveform = self.load(audio, sample_rate=sample_rate)[0]
        label = self._resolve_emotion(emotion, waveform)
        xvector = self.speaker_encoder.encode(waveform, self.sample_rate).values
        pseudo = self.speaker_generator.anonymize(
            xvector, w=w, steps=steps, generator=self._noise_generator(seed, waveform)
        )
        return self.vocoder(waveform, self.emotion_compensator(pseudo, label))

    @torch.inference_mode()
    def resynthesize(self, audio, *, sample_rate: int | None = None) -> torch.Tensor:
        """Vocoder-only baseline: resynthesize with the input's own x-vector."""
        waveform = self.load(audio, sample_rate=sample_rate)[0]
        xvector = self.speaker_encoder.encode(waveform, self.sample_rate).values
        return self.vocoder(waveform, xvector)

    def _sample_emotion(self, sample) -> dict[str, str]:
        if self.emotion_resource not in sample.resources.keys():
            return {}
        value = sample.resources[self.emotion_resource].value
        if value is None:
            raise ValueError(f"Resource {self.emotion_resource!r} is not loaded for sample {sample.utt_id!r}.")
        return {"emotion": str(value).strip()}

    def anonymize_batch(self, batch: AudioBatch, **kwargs) -> GeneratedAudio:
        """Anonymize a batch, reading per-sample labels from ``emotion_resource``."""
        sample_kwargs = None if "emotion" in kwargs else self._sample_emotion
        return self._generate_batch(batch, self.anonymize, sample_kwargs=sample_kwargs, **kwargs)
