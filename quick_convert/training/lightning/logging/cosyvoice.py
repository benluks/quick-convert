"""CosyVoice reconstruction validation media."""

from quick_convert.components.conditioning.cosyvoice_training import training_reference_kwargs
from quick_convert.components.generators.acoustic import ConditioningStream

from .media_logger import ReconstructedAudio


def cosyvoice_validation_media(system, batch, output, *, reference_seconds=None):
    stream_kwargs = {}
    if getattr(system.decoder, "stream_conditioning", None) is not None:
        stream_kwargs = training_reference_kwargs(
            system,
            batch,
            None
            if getattr(system.decoder, "content_encoder", None) is not None
            else system.decoder.flow.output_lengths(output.lengths),
            reference_seconds=reference_seconds,
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
