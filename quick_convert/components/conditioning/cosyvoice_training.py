"""Reference prefix preparation for CosyVoice training."""

import torch

from quick_convert.components.conditioning.paired_audio import paired_audio
from quick_convert.components.generators.acoustic import ConditioningStream


def training_reference_kwargs(system, batch, target_lengths=None, *, reference_seconds=None):
    """Extract prefixes from selected references, retaining original targets."""
    if reference_seconds is None:
        return {"streams": system.conditioning_streams(batch, target_lengths)}
    if batch.sample_rates is None or torch.any(batch.sample_rates != batch.sample_rates[0]):
        raise ValueError("Reference conditioning requires a common source sample rate.")
    sample_rate = int(batch.sample_rates[0])
    max_samples = int(sample_rate * reference_seconds)
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
