from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from uuid import uuid4

import torch
import torchaudio
from torch import nn

from quick_convert.components.decoders import (
    ConditioningStream,
    CosyVoiceGenerationOutput,
    CosyVoiceSpectrogramGenerator,
)
from quick_convert.components.layers.rvq import BaseResidualVectorQuantizer, RVQOutput
from quick_convert.components.mixins.resource import OnlineResourceMixin
from quick_convert.components.speaker.aggregation import SpeakerEmbeddingAggregation, audio_durations
from quick_convert.data import AudioBatch, GeneratedAudio
from quick_convert.systems.anonymization.targets import PreparedTarget, ReferenceSpeech
from quick_convert.systems.base import InferenceSystem
from quick_convert.systems.reference import ReferenceCondition, prepare_reference_audio
from quick_convert.types import AudioInput
from quick_convert.utils.audio import load_audio_input
from quick_convert.utils.masking import make_padding_mask


@dataclass
class SSLReconstructionResult:
    """Generated speech and the representations used to produce it."""

    generation: CosyVoiceGenerationOutput
    features: torch.Tensor
    lengths: torch.Tensor
    speaker_embedding: torch.Tensor | None
    encoder_output: RVQOutput | None = None


class SSLReconstructionSystem(OnlineResourceMixin, InferenceSystem):
    """Inference-ready SSL speech reconstruction independent of Lightning."""

    def __init__(
        self,
        decoder: CosyVoiceSpectrogramGenerator,
        feature_transform: nn.Module | None = None,
        *,
        online_encoders: dict[str, nn.Module] | None = None,
        trainable_online_encoders: tuple[str, ...] = (),
        encoder: BaseResidualVectorQuantizer | None = None,
    ) -> None:
        super().__init__()
        self._target_space = str(uuid4())
        self.encoder = encoder
        self.decoder = decoder
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

    @torch.inference_mode()
    def prepare_target(self, target: ReferenceSpeech | PreparedTarget) -> PreparedTarget:
        """Prepare the saved decoder's native reference or named streams."""
        if self.training:
            raise RuntimeError("Call eval() before preparing targets.")
        conditioning = getattr(self.decoder, "stream_conditioning", None)
        needs_embedding = self.decoder.requires_speaker_embedding
        if isinstance(target, PreparedTarget):
            if target.space != self.target_space or target.representation != "ssl-reference-v1":
                raise ValueError("Prepared target belongs to a different model or representation.")
            if not isinstance(target.conditioning, ReferenceCondition):
                raise ValueError("Expected prepared reference conditioning.")
            probe = next(self.parameters(), torch.empty(0))
            kwargs = target.conditioning.decoder_kwargs(1, device=probe.device, dtype=probe.dtype)
            return replace(
                target,
                conditioning=ReferenceCondition(
                    kwargs["speaker_embedding"],
                    kwargs.get("reference"),
                    kwargs.get("streams"),
                ),
            )
        if not isinstance(target, ReferenceSpeech):
            raise TypeError("Expected ReferenceSpeech or PreparedTarget.")
        if not needs_embedding and target.aggregation is not None:
            raise ValueError("Stream targets without a global embedding do not support embedding aggregation.")
        if target.allow_missing_streams and conditioning is None:
            raise ValueError("allow_missing_streams requires named stream conditioning.")
        if target.prompt_seconds is None:
            raise ValueError("Reference targets require prompt_seconds.")
        sample_rate = self.online_encoders["content"].sample_rate
        paths = tuple(sorted(target.paths)) if target.prompt_builder is not None else target.paths

        def load_batch(path):
            waveform = load_audio_input(path, target_sample_rate=sample_rate, mono=True).reshape(-1)
            return self._reference_batch(waveform, sample_rate)

        embedding = None
        durations = []
        aggregation = None
        if needs_embedding:
            policy = target.aggregation or SpeakerEmbeddingAggregation()
            embeddings = []
            for path in paths:
                recording = load_batch(path)
                durations.extend(audio_durations(recording))
                embeddings.append(self.speaker_condition(recording))
            embedding = policy(torch.cat(embeddings), durations=durations).unsqueeze(0).detach().clone()
            aggregation = policy.method
        batch, prompt = prepare_reference_audio(target, load_batch)
        streams = None
        reference = None
        if conditioning is None:
            content = self.get_resource(batch, "content")
            if content.lengths is None:
                raise ValueError("Reference content requires valid lengths.")
            features, lengths, _ = self.encode_features(content.values, content.lengths)
            reference = self.decoder.prepare_reference(
                features,
                lengths,
                batch.waveforms,
                batch.lengths,
                sample_rate,
            )
        else:
            lengths = self.decoder.mel_output_lengths(batch.lengths, sample_rate)
            if torch.any(lengths <= 0):
                raise ValueError("Reference prompt is too short for the mel clock.")
            streams = {}
            for name in conditioning.dimensions:
                if name == "mel":
                    mels, mel_lengths = self.decoder._compute_mels(batch.waveforms, batch.lengths, sample_rate)
                    stream = ConditioningStream(mels.transpose(1, 2), mel_lengths, True)
                else:
                    encoder = self.online_encoders[name] if name in self.online_encoders else None
                    encode = getattr(encoder, "encode_prefixes", None)
                    if not callable(encode):
                        message = f"No reference prefix extractor configured for stream {name!r}."
                        if not target.allow_missing_streams:
                            raise ValueError(message)
                        warnings.warn(message + " Continuing without this stream.", RuntimeWarning, stacklevel=2)
                        continue
                    if encoder.sample_rate != sample_rate:
                        raise ValueError(f"Stream {name!r} requires the content sample rate.")
                    try:
                        resource = encode(
                            batch, lengths, self.decoder.mel_config.sample_rate / self.decoder.mel_config.hop_size
                        )
                    except ImportError as error:
                        if not target.allow_missing_streams:
                            raise
                        warnings.warn(f"Skipping unavailable stream {name!r}: {error}", RuntimeWarning, stacklevel=2)
                        continue
                    stream = ConditioningStream(resource.values, resource.lengths, True)
                streams[name] = stream
        return PreparedTarget(
            ReferenceCondition(embedding, reference, streams),
            self.target_space,
            "ssl-reference-v1",
            paths,
            durations=tuple(durations),
            aggregation=aggregation,
            prompt_segments=prompt.segments if prompt is not None else (),
            prompt_sample_rate=sample_rate,
            prompt_crossfade_seconds=target.prompt_builder.crossfade_seconds if prompt is not None else None,
        )

    @torch.inference_mode()
    def convert_batch(
        self, batch: AudioBatch, *, target: PreparedTarget, inference_cfg_rate: float | None = None
    ) -> GeneratedAudio:
        """Generate source content with the prepared target conditions."""
        if self.training:
            raise RuntimeError("Call eval() before converting speech.")
        target = self.prepare_target(target)
        content = self.get_resource(batch, "content")
        if content.lengths is None:
            raise ValueError("Content features require valid lengths.")
        features, lengths, _ = self.encode_features(content.values, content.lengths)
        kwargs = target.conditioning.decoder_kwargs(len(batch), device=features.device, dtype=features.dtype)
        generation = self.decoder(
            feature=features,
            length=lengths,
            run_vocoder=True,
            **kwargs,
            **({"inference_cfg_rate": inference_cfg_rate} if inference_cfg_rate is not None else {}),
        )
        if generation.audio is None:
            raise RuntimeError("Conversion requires generated audio.")
        return generation.audio

    def train(self, mode: bool = True):
        super().train(mode)
        for name, encoder in self.online_encoders.items():
            if name not in self.trainable_online_encoders:
                encoder.eval()
        vocoder = getattr(self.decoder, "vocoder", None)
        if vocoder is not None and not any(parameter.requires_grad for parameter in vocoder.parameters()):
            vocoder.eval()
        return self

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
        speaker_embedding: torch.Tensor | None = None,
        *,
        run_vocoder: bool = True,
        streams: dict[str, ConditioningStream] | None = None,
    ) -> SSLReconstructionResult:
        """Generate from plain content and speaker tensors."""
        features, output_lengths, encoder_output = self.encode_features(values, lengths)
        generation = self.decode_features(
            features,
            output_lengths,
            speaker_embedding,
            run_vocoder=run_vocoder,
            **({"streams": streams} if streams is not None else {}),
        )
        return SSLReconstructionResult(
            generation=generation,
            features=features,
            lengths=output_lengths,
            speaker_embedding=speaker_embedding,
            encoder_output=encoder_output,
        )

    def decode_features(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        speaker_embedding: torch.Tensor | None = None,
        *,
        run_vocoder: bool = True,
        streams: dict[str, ConditioningStream] | None = None,
    ) -> CosyVoiceGenerationOutput:
        """Generate from already transformed or encoded content features."""
        return self.decoder(
            feature=features,
            length=lengths,
            speaker_embedding=speaker_embedding,
            run_vocoder=run_vocoder,
            **({"streams": streams} if streams is not None else {}),
        )

    def speaker_condition(self, batch: AudioBatch) -> torch.Tensor | None:
        """Resolve the global speaker resource only when the decoder consumes it."""
        if not getattr(self.decoder, "requires_speaker_embedding", True):
            return None
        return self.get_resource(batch, "speaker").values

    def conditioning_streams(
        self, batch: AudioBatch, target_lengths: torch.Tensor | None = None
    ) -> dict[str, ConditioningStream]:
        """Resolve training streams from named resources already on the mel clock."""
        conditioning = getattr(self.decoder, "stream_conditioning", None)
        if conditioning is None:
            return {}
        streams = {}
        for name in conditioning.dimensions:
            if name == "mel":
                continue
            supplied = batch.resources.get(name)
            if isinstance(supplied, ConditioningStream):
                streams[name] = supplied
                continue
            encoder = self.online_encoders[name] if name in self.online_encoders else None
            encode_prefixes = getattr(encoder, "encode_prefixes", None)
            if supplied is None and callable(encode_prefixes):
                mel_lengths = self.decoder.mel_output_lengths(batch.lengths, int(batch.sample_rates[0]))
                if target_lengths is not None:
                    mel_lengths = torch.minimum(mel_lengths, target_lengths)
                ratio = self.decoder.flow.token_mel_ratio
                mel_lengths = mel_lengths // ratio * ratio
                prefix_lengths = conditioning.sample_prefix_lengths(mel_lengths)
                with torch.no_grad():
                    resource = encode_prefixes(
                        batch,
                        prefix_lengths,
                        self.decoder.mel_config.sample_rate / self.decoder.mel_config.hop_size,
                    )
                resource = self._normalize_resource(resource)
                streams[name] = ConditioningStream(resource.values, resource.lengths, is_prefix=True)
                continue
            resource = self.get_resource(batch, name)
            if resource.lengths is None:
                raise ValueError(f"Frame-wise conditioning resource {name!r} requires lengths.")
            streams[name] = ConditioningStream(resource.values, resource.lengths)
        return streams

    @staticmethod
    def save_generation(generation: CosyVoiceGenerationOutput, path: PathLike) -> None:
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
        self, batch: AudioBatch, *, run_vocoder: bool = True, streams: dict[str, ConditioningStream] | None = None
    ) -> SSLReconstructionResult:
        content = self.get_resource(batch, "content")
        if content.lengths is None:
            raise ValueError("Content features require valid lengths for SSL reconstruction.")
        speaker_embedding = self.speaker_condition(batch)
        return self.generate_features(
            content.values,
            content.lengths,
            speaker_embedding,
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
        streams: dict[str, ConditioningStream] | None = None,
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
