from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from quick_convert.components.decoders import CosyVoiceGenerationOutput, CosyVoiceSpectrogramGenerator
from quick_convert.components.layers.rvq import BaseResidualVectorQuantizer, RVQOutput
from quick_convert.components.mixins.resource import OnlineResourceMixin
from quick_convert.data import AudioBatch
from quick_convert.types import AudioInput
from quick_convert.utils.audio import load_audio_input
from quick_convert.utils.masking import make_padding_mask


@dataclass
class SSLReconstructionResult:
    """Generated speech and the representations used to produce it."""

    generation: CosyVoiceGenerationOutput
    features: torch.Tensor
    lengths: torch.Tensor
    speaker_embedding: torch.Tensor
    encoder_output: RVQOutput | None = None


class SSLReconstructionSystem(OnlineResourceMixin, nn.Module):
    """Inference-ready SSL speech reconstruction independent of Lightning."""

    def __init__(
        self,
        decoder: CosyVoiceSpectrogramGenerator,
        feature_transform: nn.Module | None = None,
        *,
        online_encoders: dict[str, nn.Module] | None = None,
        encoder: BaseResidualVectorQuantizer | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.feature_transform = feature_transform or nn.Identity()
        self.online_encoders = nn.ModuleDict(online_encoders or {})

        self.online_encoders.requires_grad_(False)
        self.online_encoders.eval()

    @staticmethod
    def _output_lengths(module: nn.Module, lengths: torch.Tensor, role: str) -> torch.Tensor:
        if isinstance(module, nn.Identity):
            return lengths
        output_lengths = getattr(module, "output_lengths", None)
        if not callable(output_lengths):
            raise TypeError(f"SSL reconstruction {role} modules must expose output_lengths(input_lengths).")
        return output_lengths(lengths)

    def encode_features(
        self,
        values: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, RVQOutput | None]:
        """Transform content tensors without requiring Quick Convert batch types."""
        features = self.feature_transform(values)
        transformed_lengths = self._output_lengths(self.feature_transform, lengths, "feature_transform")

        if transformed_lengths.shape != (features.shape[0],):
            raise ValueError(
                f"Expected one content length per batch item, got {tuple(transformed_lengths.shape)} "
                f"for batch size {features.shape[0]}."
            )
        if torch.any(transformed_lengths > features.shape[1]):
            raise ValueError(
                "A transformed content length exceeds the padded feature time dimension: "
                f"maximum is {features.shape[1]}, got {int(transformed_lengths.max())}."
            )

        if self.encoder is None:
            return features, transformed_lengths, None

        padding_mask = make_padding_mask(transformed_lengths, max_length=features.shape[1])
        encoder_output = self.encoder(features.transpose(1, 2), padding_mask)
        encoded_lengths = self.encoder.output_lengths(transformed_lengths)

        if torch.any(encoded_lengths > encoder_output.z_q.shape[2]):
            raise RuntimeError(
                "An encoder output length exceeds the quantized time dimension: "
                f"maximum is {encoder_output.z_q.shape[2]}, got {int(encoded_lengths.max())}."
            )

        return encoder_output.z_q.transpose(1, 2), encoded_lengths, encoder_output

    def generate_features(
        self,
        values: torch.Tensor,
        lengths: torch.Tensor,
        speaker_embedding: torch.Tensor,
        *,
        run_vocoder: bool = True,
    ) -> SSLReconstructionResult:
        """Generate from plain content and speaker tensors."""
        features, output_lengths, encoder_output = self.encode_features(values, lengths)
        generation = self.decoder(
            feature=features,
            length=output_lengths,
            speaker_embedding=speaker_embedding,
            run_vocoder=run_vocoder,
        )
        return SSLReconstructionResult(
            generation=generation,
            features=features,
            lengths=output_lengths,
            speaker_embedding=speaker_embedding,
            encoder_output=encoder_output,
        )

    def forward(self, batch: AudioBatch, *, run_vocoder: bool = True) -> SSLReconstructionResult:
        content = self.get_resource(batch, "content")
        if content.lengths is None:
            raise ValueError("Content features require valid lengths for SSL reconstruction.")
        speaker_embedding = self.get_resource(batch, "speaker").values
        return self.generate_features(
            content.values,
            content.lengths,
            speaker_embedding,
            run_vocoder=run_vocoder,
        )

    @torch.inference_mode()
    def reconstruct(
        self,
        audio: AudioInput,
        *,
        sample_rate: int | None = None,
        run_vocoder: bool = True,
    ) -> SSLReconstructionResult:
        """Reconstruct a single audio file or in-memory waveform tensor."""
        if "content" not in self.online_encoders or "speaker" not in self.online_encoders:
            raise RuntimeError("Audio reconstruction requires `content` and `speaker` online encoders.")

        content_encoder = self.online_encoders["content"]
        speaker_encoder = self.online_encoders["speaker"]
        content_sample_rate = getattr(content_encoder, "sample_rate", None)
        speaker_sample_rate = getattr(speaker_encoder, "sample_rate", None)
        if content_sample_rate is None or content_sample_rate != speaker_sample_rate:
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
        return self(batch, run_vocoder=run_vocoder)
