"""CosyVoice-specific flow objective and validation media."""

from __future__ import annotations

import torch

from quick_convert.components.conditioning.paired_audio import paired_audio
from quick_convert.components.generators.acoustic import ConditioningStream
from quick_convert.training.lightning.logging.media_logger import ReconstructedAudio


class CosyVoiceFlowObjective:
    """Legacy flow loss and reference-prefix policy."""

    def __init__(self, reference_seconds=None):
        self.reference_seconds = reference_seconds

    def conditioning_kwargs(self, system, batch, target_lengths=None):
        """Extract prefixes from selected references, retaining original targets."""
        if self.reference_seconds is None:
            return {"streams": system.conditioning_streams(batch, target_lengths)}
        if batch.sample_rates is None or torch.any(batch.sample_rates != batch.sample_rates[0]):
            raise ValueError("Reference conditioning requires a common source sample rate.")
        sample_rate = int(batch.sample_rates[0])
        max_samples = int(sample_rate * self.reference_seconds)
        if max_samples < 1:
            raise ValueError("reference_seconds must contain at least one sample.")
        reference = paired_audio(batch, "reference_path", sample_rate, max_samples)
        streams = system.conditioning_streams(reference)
        if "mel" in system.decoder.stream_conditioning.dimensions:
            mels, mel_lengths = system.decoder._compute_mels(reference.waveforms, reference.lengths, sample_rate)
            prefix_lengths = system.decoder.stream_conditioning.sample_prefix_lengths(mel_lengths)
            streams["mel"] = ConditioningStream(
                mels.transpose(1, 2)[:, : int(prefix_lengths.max())], prefix_lengths, is_prefix=True
            )
        content = system.get_resource(reference, "content")
        if content.lengths is None:
            raise ValueError("Reference content requires lengths.")
        features, lengths, _ = system.encode_features(content.values, content.lengths, metadata=content.metadata)
        return {"streams": streams, "prefix_content": (features, lengths)}

    def compute_loss(self, system, batch, features, lengths, speaker_embedding):
        decoder_output = system.decoder.compute_loss(
            features=features,
            lengths=lengths,
            target_wav=batch.waveforms,
            wav_lens=batch.lengths,
            sampling_rate=batch.sample_rates[0],
            speaker_embedding=speaker_embedding,
            **(
                self.conditioning_kwargs(
                    system,
                    batch,
                    None
                    if getattr(system.decoder, "content_encoder", None) is not None
                    else system.decoder.flow.output_lengths(lengths),
                )
                if getattr(system.decoder, "stream_conditioning", None) is not None
                else {}
            ),
        )

        return decoder_output

    def validation_media(self, system, batch, output):
        stream_kwargs = {}
        if getattr(system.decoder, "stream_conditioning", None) is not None:
            stream_kwargs = self.conditioning_kwargs(
                system,
                batch,
                None
                if getattr(system.decoder, "content_encoder", None) is not None
                else system.decoder.flow.output_lengths(output.lengths),
            )
            streams = stream_kwargs["streams"]
            if "mel" in system.decoder.stream_conditioning.dimensions and "mel" not in streams:
                mels, mel_lengths = system.decoder._compute_mels(
                    batch.waveforms, batch.lengths, sampling_rate=batch.sample_rates[0]
                )
                streams["mel"] = ConditioningStream(mels.transpose(1, 2), mel_lengths)
            # Validation exercises prefix-only inference rather than leaking target streams.
            prefixes = {}
            for name, stream in streams.items():
                if stream.is_prefix:
                    prefixes[name] = stream
                    continue
                lengths = (stream.lengths // 5).clamp_min(1)
                prefixes[name] = ConditioningStream(stream.values[:, : int(lengths.max())], lengths)
            stream_kwargs["streams"] = prefixes
        if getattr(system.decoder, "content_encoder", None) is not None:
            target_lengths = system.decoder.mel_output_lengths(batch.lengths, int(batch.sample_rates[0]))
            stream_kwargs["duration_ratio"] = target_lengths / system.decoder.flow.output_lengths(output.lengths)
        generation = system.decode_features(
            output.features,
            output.lengths,
            output.speaker_embedding,
            run_vocoder=True,
            **stream_kwargs,
        )
        if generation.audio is None:
            raise RuntimeError("Decoder did not return audio when run_vocoder=True.")
        original_mel, original_mel_lengths = system.decoder._compute_mels(
            batch.waveforms, batch.lengths, sampling_rate=batch.sample_rates[0]
        )
        return ReconstructedAudio(
            original_audio=batch.waveforms,
            reconstructed_audio=generation.audio.waveforms,
            original_mel=original_mel,
            reconstructed_mel=generation.mel,
            original_audio_lengths=batch.lengths,
            reconstructed_audio_lengths=generation.audio.lengths,
            original_mel_lengths=original_mel_lengths,
            reconstructed_mel_lengths=generation.mel_lengths,
            original_sample_rate=batch.sample_rates[0],
            reconstructed_sample_rate=generation.audio.sample_rate,
            ids=batch.utt_ids,
        )
