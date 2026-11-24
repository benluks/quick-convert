from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
import torchaudio
from torch import nn

from quick_convert.components.conditioning.cosyvoice_reconstruction import CosyVoiceReconstruction
from quick_convert.components.layers.rvq import BaseResidualVectorQuantizer, RVQOutput
from quick_convert.components.resources.resolution import OnlineResourceMixin
from quick_convert.data import AudioBatch
from quick_convert.systems._content import content_frame_rate, quantize_content, transform_content
from quick_convert.systems.base import InferenceSystem
from quick_convert.types import AudioInput
from quick_convert.utils.audio import load_audio_input


@dataclass
class SSLReconstructionResult:
    """Generated speech and the representations used to produce it."""

    generation: Any
    features: torch.Tensor
    lengths: torch.Tensor
    speaker_embedding: torch.Tensor | None
    encoder_output: RVQOutput | None = None


class SSLReconstructionConversionSystem(OnlineResourceMixin, InferenceSystem):
    """Inference-ready SSL speech reconstruction independent of Lightning."""

    def __init__(
        self,
        decoder: nn.Module,
        feature_transform: nn.Module | None = None,
        *,
        online_encoders: dict[str, nn.Module] | None = None,
        trainable_online_encoders: tuple[str, ...] = (),
        encoder: BaseResidualVectorQuantizer | None = None,
        reconstruction=None,
    ) -> None:
        super().__init__()
        self._target_space = str(uuid4())
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction = reconstruction if reconstruction is not None else CosyVoiceReconstruction()
        self.feature_transform = feature_transform or nn.Identity()
        self.configure_online_encoders(
            {name: encoder for name, encoder in (online_encoders or {}).items() if encoder is not None},
            trainable_online_encoders,
        )

    @property
    def target_space(self) -> str:
        """Identity of the model that owns prepared prefix streams."""
        return self._target_space

    def _reference_batch(self, waveform, sample_rate):
        device = next(self.parameters(), torch.empty(0)).device
        return AudioBatch(
            utt_ids=["reference"],
            paths=[Path("reference")],
            splits=[None],
            resources={},
            waveforms=waveform.to(device).unsqueeze(0),
            lengths=torch.tensor([waveform.numel()], device=device),
            sample_rates=torch.tensor([sample_rate], device=device),
        )

    def prepare_target(self, target):
        return self.reconstruction.prepare_target(self, target)

    def convert_batch(self, batch, *, target, **kwargs):
        return self.reconstruction.convert_batch(self, batch, target=target, **kwargs)

    def decode_features(self, features, lengths, speaker_embedding=None, **kwargs):
        return self.reconstruction.decode_features(self, features, lengths, speaker_embedding, **kwargs)

    def speaker_condition(self, batch):
        return self.reconstruction.speaker_condition(self, batch)

    def conditioning_streams(self, batch, target_lengths=None):
        return self.reconstruction.conditioning_streams(self, batch, target_lengths)

    def train(self, mode: bool = True):
        super().train(mode)
        for name, encoder in self.online_encoders.items():
            if name not in self.trainable_online_encoders:
                encoder.eval()
        vocoder = getattr(self.decoder, "vocoder", None)
        if vocoder is not None and not any(parameter.requires_grad for parameter in vocoder.parameters()):
            vocoder.eval()
        return self

    def encode_features(
        self,
        values: torch.Tensor,
        lengths: torch.Tensor,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RVQOutput | None]:
        """Adapt continuous content to decoder features, optionally checking source metadata."""
        rate = content_frame_rate(metadata, self.feature_transform, self.encoder)
        expected_rate = self.reconstruction.input_frame_rate(self)
        if rate is not None and expected_rate is not None and not math.isclose(rate, expected_rate, rel_tol=1e-6):
            raise ValueError(
                f"Adapted content frame rate {rate} differs from decoder input frame rate {expected_rate}."
            )
        features, transformed_lengths = transform_content(values, lengths, self.feature_transform, metadata=metadata)
        if self.encoder is None:
            return features, transformed_lengths, None
        output, encoded_lengths = quantize_content(features, transformed_lengths, self.encoder)
        return output.z_q.transpose(1, 2), encoded_lengths, output

    def generate_features(
        self,
        values: torch.Tensor,
        lengths: torch.Tensor,
        speaker_embedding: torch.Tensor | None = None,
        *,
        run_vocoder: bool = True,
        streams: dict[str, Any] | None = None,
        duration_ratio: float | torch.Tensor = 1.0,
        content_metadata: Mapping[str, Any] | None = None,
    ) -> SSLReconstructionResult:
        """Generate from plain content and speaker tensors."""
        features, output_lengths, encoder_output = self.encode_features(values, lengths, metadata=content_metadata)
        generation = self.decode_features(
            features,
            output_lengths,
            speaker_embedding,
            run_vocoder=run_vocoder,
            duration_ratio=duration_ratio,
            **({"streams": streams} if streams is not None else {}),
        )
        return SSLReconstructionResult(
            generation=generation,
            features=features,
            lengths=output_lengths,
            speaker_embedding=speaker_embedding,
            encoder_output=encoder_output,
        )

    @staticmethod
    def save_generation(generation: Any, path: PathLike) -> None:
        """Save a single generated waveform."""
        if generation.audio is None:
            raise ValueError("Saving generated audio requires run_vocoder=True.")
        if len(generation.audio) != 1:
            raise ValueError("Saving generated audio only supports a single waveform.")
        torchaudio.save(
            Path(path),
            generation.audio.waveform(0).unsqueeze(0).to("cpu"),
            generation.audio.sample_rate,
        )

    def forward(
        self, batch: AudioBatch, *, run_vocoder: bool = True, streams: dict[str, Any] | None = None
    ) -> SSLReconstructionResult:
        content = self.get_resource(batch, "content")
        if content.lengths is None:
            raise ValueError("Content features require valid lengths for SSL reconstruction.")
        speaker_embedding = self.speaker_condition(batch)
        return self.generate_features(
            content.values,
            content.lengths,
            speaker_embedding,
            content_metadata=content.metadata,
            run_vocoder=run_vocoder,
            **({"streams": streams} if streams is not None else {}),
        )

    @torch.inference_mode()
    def reconstruct(
        self,
        audio: AudioInput,
        *,
        sample_rate: int | None = None,
        run_vocoder: bool = True,
        to_file: PathLike | None = None,
        streams: dict[str, Any] | None = None,
    ) -> SSLReconstructionResult:
        """Reconstruct a single audio file or in-memory waveform tensor."""
        if "content" not in self.online_encoders:
            raise RuntimeError("Audio reconstruction requires a `content` online encoder.")
        content_encoder = self.online_encoders["content"]
        content_sample_rate = getattr(content_encoder, "sample_rate", None)
        if content_sample_rate is None:
            raise ValueError("Content encoder must declare its input sample rate.")
        if getattr(self.decoder, "requires_speaker_embedding", True):
            if "speaker" not in self.online_encoders:
                raise RuntimeError("Audio reconstruction requires a `speaker` online encoder.")
            speaker_sample_rate = getattr(self.online_encoders["speaker"], "sample_rate", None)
            if content_sample_rate != speaker_sample_rate:
                raise ValueError("Content and speaker encoders must declare the same input sample rate.")

        device = getattr(content_encoder, "device", torch.device("cpu"))
        waveform = load_audio_input(
            audio,
            target_sample_rate=content_sample_rate,
            sample_rate=sample_rate,
            mono=True,
            device=device,
        ).squeeze(0)
        batch = AudioBatch(
            utt_ids=[Path(audio).stem if not isinstance(audio, torch.Tensor) else "audio"],
            paths=[Path(audio) if not isinstance(audio, torch.Tensor) else Path("audio")],
            splits=[None],
            resources={},
            waveforms=waveform.unsqueeze(0),
            lengths=torch.tensor([waveform.shape[-1]], device=device),
            sample_rates=torch.tensor([content_sample_rate], device=device),
        )
        self.eval()
        result = self(batch, run_vocoder=run_vocoder, **({"streams": streams} if streams is not None else {}))
        if to_file is not None:
            self.save_generation(result.generation, to_file)
        return result
