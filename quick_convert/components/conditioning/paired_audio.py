"""Resolve paired audio resources."""

from dataclasses import replace
from pathlib import Path

import torch

from quick_convert.data import AudioBatch


def paired_audio(batch, name, sample_rate, max_samples=None):
    """Apply pairing and prefix policy to an already loaded audio resource."""
    audio = batch.resources.get(name)
    if not isinstance(audio, AudioBatch) or audio.waveforms is None or len(audio) != len(batch):
        raise ValueError(f"Paired audio requires a loaded {name!r} audio resource per example.")
    if audio.lengths is None or audio.sample_rates is None or torch.any(audio.sample_rates != sample_rate):
        raise ValueError(f"Configure audio resource {name!r} with sample_rate={sample_rate}.")
    if any(
        Path(path).resolve() == Path(target).resolve() for path, target in zip(audio.paths, batch.paths, strict=True)
    ):
        raise ValueError(f"The {name} recording must differ from the target recording.")
    device = batch.waveforms.device
    return replace(
        audio,
        waveforms=audio.waveforms[:, :max_samples].to(device),
        lengths=(audio.lengths if max_samples is None else audio.lengths.clamp_max(max_samples)).to(device),
        sample_rates=audio.sample_rates.to(device),
    )
