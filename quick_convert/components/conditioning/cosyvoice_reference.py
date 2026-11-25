"""CosyVoice-specific reference preparation for SSL reconstruction."""

from __future__ import annotations

import warnings
from dataclasses import replace

import torch

from quick_convert.components.generators.acoustic import ConditioningStream
from quick_convert.components.transforms.speaker_aggregation import SpeakerEmbeddingAggregation, audio_durations
from quick_convert.data import AudioBatch, GeneratedAudio
from quick_convert.systems.conversion.reference import ReferenceCondition, prepare_reference_audio
from quick_convert.systems.conversion.targets import PreparedTarget, ReferenceSpeech, SourceVoice
from quick_convert.utils.audio import load_audio_input


class CosyVoiceReferenceConditioning:
    """CosyVoice reference conditioning."""

    @staticmethod
    def _validate_prepared_target(system, target: PreparedTarget) -> None:
        if target.representation == "ssl-source-v1":
            if target.space != system.target_space or target.conditioning is not None:
                raise ValueError("Prepared source target belongs to a different model or representation.")
            return
        if target.space != system.target_space or target.representation != "ssl-reference-v1":
            raise ValueError("Prepared target belongs to a different model or representation.")
        if not isinstance(target.conditioning, ReferenceCondition):
            raise ValueError("Expected prepared reference conditioning.")
        if (
            getattr(system.decoder, "stream_conditioning", None) is not None
            and target.conditioning.prefix_content is None
        ):
            raise ValueError(
                "Prepared stream reference requires prefix content. Prepare it again from ReferenceSpeech."
            )

    @staticmethod
    @torch.inference_mode()
    def prepare_target(system, target: ReferenceSpeech | SourceVoice | PreparedTarget) -> PreparedTarget:
        """Prepare the saved decoder's native reference or named streams."""
        if system.training:
            raise RuntimeError("Call eval() before preparing targets.")
        if isinstance(target, SourceVoice):
            return PreparedTarget(None, system.target_space, "ssl-source-v1")
        if isinstance(target, PreparedTarget):
            CosyVoiceReferenceConditioning._validate_prepared_target(system, target)
            if target.representation == "ssl-source-v1":
                return target
            probe = next(system.parameters(), torch.empty(0))
            kwargs = target.conditioning.decoder_kwargs(1, device=probe.device, dtype=probe.dtype)
            return replace(
                target,
                conditioning=ReferenceCondition(
                    kwargs["speaker_embedding"],
                    kwargs.get("reference"),
                    kwargs.get("streams"),
                    kwargs.get("prefix_content"),
                ),
            )
        if not isinstance(target, ReferenceSpeech):
            raise TypeError("Expected ReferenceSpeech or PreparedTarget.")
        conditioning = getattr(system.decoder, "stream_conditioning", None)
        needs_embedding = system.decoder.requires_speaker_embedding
        if not needs_embedding and target.aggregation is not None:
            raise ValueError("Stream targets without a global embedding do not support embedding aggregation.")
        if target.allow_missing_streams and conditioning is None:
            raise ValueError("allow_missing_streams requires named stream conditioning.")
        if target.prompt_seconds is None:
            raise ValueError("Reference targets require prompt_seconds.")
        sample_rate = system.online_encoders["content"].sample_rate
        paths = tuple(sorted(target.paths)) if target.prompt_builder is not None else target.paths

        def load_batch(path):
            waveform = load_audio_input(path, target_sample_rate=sample_rate, mono=True).reshape(-1)
            return system._reference_batch(waveform, sample_rate)

        embedding = None
        durations = []
        aggregation = None
        if needs_embedding:
            policy = target.aggregation or SpeakerEmbeddingAggregation()
            embeddings = []
            for path in paths:
                recording = load_batch(path)
                durations.extend(audio_durations(recording))
                embeddings.append(system.speaker_condition(recording))
            embedding = policy(torch.cat(embeddings), durations=durations).unsqueeze(0).detach().clone()
            aggregation = policy.method
        batch, prompt = prepare_reference_audio(target, load_batch)
        content = system.get_resource(batch, "content")
        if content.lengths is None:
            raise ValueError("Reference content requires valid lengths.")
        features, content_lengths, _ = system.encode_features(
            content.values, content.lengths, metadata=content.metadata
        )
        streams = None
        reference = None
        prefix_content = None
        if conditioning is None:
            reference = system.decoder.prepare_reference(
                features,
                content_lengths,
                batch.waveforms,
                batch.lengths,
                sample_rate,
            )
        else:
            prefix_content = (features, content_lengths)
            lengths = system.decoder.mel_output_lengths(batch.lengths, sample_rate)
            if torch.any(lengths <= 0):
                raise ValueError("Reference prompt is too short for the mel clock.")
            streams = {}
            for name in conditioning.dimensions:
                if name == "mel":
                    mels, mel_lengths = system.decoder._compute_mels(batch.waveforms, batch.lengths, sample_rate)
                    stream = ConditioningStream(mels.transpose(1, 2), mel_lengths, True)
                else:
                    encoder = system.online_encoders[name] if name in system.online_encoders else None
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
                            batch, lengths, system.decoder.mel_config.sample_rate / system.decoder.mel_config.hop_size
                        )
                    except ImportError as error:
                        if not target.allow_missing_streams:
                            raise
                        warnings.warn(f"Skipping unavailable stream {name!r}: {error}", RuntimeWarning, stacklevel=2)
                        continue
                    stream = ConditioningStream(resource.values, resource.lengths, True)
                streams[name] = stream
        return PreparedTarget(
            ReferenceCondition(embedding, reference, streams, prefix_content),
            system.target_space,
            "ssl-reference-v1",
            paths,
            durations=tuple(durations),
            aggregation=aggregation,
            prompt_segments=prompt.segments if prompt is not None else (),
            prompt_sample_rate=sample_rate,
            prompt_crossfade_seconds=target.prompt_builder.crossfade_seconds if prompt is not None else None,
        )

    @staticmethod
    @torch.inference_mode()
    def convert_batch(
        system, batch: AudioBatch, *, target: PreparedTarget, inference_cfg_rate: float | None = None
    ) -> GeneratedAudio:
        """Generate source content with the prepared target conditions."""
        if system.training:
            raise RuntimeError("Call eval() before converting speech.")
        if isinstance(target, PreparedTarget):
            CosyVoiceReferenceConditioning._validate_prepared_target(system, target)
        else:
            target = system.prepare_target(target)
        if target.representation == "ssl-source-v1":
            if inference_cfg_rate is not None:
                raise ValueError("SourceVoice does not accept an inference_cfg_rate override.")
            generation = system(batch, run_vocoder=True).generation
            if generation.audio is None:
                raise RuntimeError("Conversion requires generated audio.")
            return generation.audio
        content = system.get_resource(batch, "content")
        if content.lengths is None:
            raise ValueError("Content features require valid lengths.")
        features, lengths, _ = system.encode_features(content.values, content.lengths, metadata=content.metadata)
        kwargs = target.conditioning.decoder_kwargs(len(batch), device=features.device, dtype=features.dtype)
        generation = system.decode_features(
            features,
            lengths,
            run_vocoder=True,
            **kwargs,
            **({"inference_cfg_rate": inference_cfg_rate} if inference_cfg_rate is not None else {}),
        )
        if generation.audio is None:
            raise RuntimeError("Conversion requires generated audio.")
        return generation.audio

    @staticmethod
    def conditioning_streams(
        system, batch: AudioBatch, target_lengths: torch.Tensor | None = None
    ) -> dict[str, ConditioningStream]:
        """Resolve training streams from named resources already on the mel clock."""
        conditioning = getattr(system.decoder, "stream_conditioning", None)
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
            encoder = system.online_encoders[name] if name in system.online_encoders else None
            encode_prefixes = getattr(encoder, "encode_prefixes", None)
            if supplied is None and callable(encode_prefixes):
                mel_lengths = system.decoder.mel_output_lengths(batch.lengths, int(batch.sample_rates[0]))
                if target_lengths is not None:
                    mel_lengths = torch.minimum(mel_lengths, target_lengths)
                ratio = system.decoder.flow.token_mel_ratio
                mel_lengths = mel_lengths // ratio * ratio
                prefix_lengths = conditioning.sample_prefix_lengths(mel_lengths)
                with torch.no_grad():
                    resource = encode_prefixes(
                        batch,
                        prefix_lengths,
                        system.decoder.mel_config.sample_rate / system.decoder.mel_config.hop_size,
                    )
                resource = system._normalize_resource(resource)
                streams[name] = ConditioningStream(resource.values, resource.lengths, is_prefix=True)
                continue
            resource = system.get_resource(batch, name)
            if resource.lengths is None:
                raise ValueError(f"Frame-wise conditioning resource {name!r} requires lengths.")
            streams[name] = ConditioningStream(resource.values, resource.lengths)
        return streams
