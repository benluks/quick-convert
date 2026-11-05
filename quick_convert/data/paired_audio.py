"""Materialize paired recordings from a batch's named path resources."""

from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from quick_convert.utils.audio import load_audio_input

from .types import AudioBatch


def load_paired_audio_batch(
    batch: AudioBatch,
    name: str,
    sample_rate: int,
    max_samples: int | None = None,
) -> AudioBatch:
    """Load a separate recording per target without modifying the target batch.

    ``batch.resources[name]`` must contain one path per example, in batch order.
    Paths must resolve to recordings other than the corresponding target.
    Audio is mixed to mono, resampled to ``sample_rate``, and optionally limited
    to its first ``max_samples`` samples after resampling. Empty recordings are
    rejected. The caller selects the resource and any duration limit.

    Return a padded batch on the target waveform device with exact lengths,
    copied utterance IDs and splits, paired paths, and no inherited resources.
    The target batch must have loaded waveforms.
    """
    paths = batch.resources.get(name)
    if paths is None or isinstance(paths, (str, Path)) or len(paths) != len(batch):
        raise ValueError(f"Paired audio requires one {name} per example.")
    waveforms = []
    for path, source in zip(paths, batch.paths, strict=True):
        if Path(path).resolve() == Path(source).resolve():
            raise ValueError(f"The {name} recording must differ from the target recording.")
        waveform = load_audio_input(path, target_sample_rate=sample_rate, mono=True)[0, :max_samples]
        if waveform.numel() == 0:
            raise ValueError(f"Paired recording {path!r} is empty.")
        waveforms.append(waveform)
    device = batch.waveforms.device
    lengths = torch.tensor([waveform.numel() for waveform in waveforms], device=device)
    return AudioBatch(
        utt_ids=list(batch.utt_ids),
        paths=[Path(path) for path in paths],
        splits=list(batch.splits),
        resources={},
        waveforms=pad_sequence(waveforms, batch_first=True).to(device),
        lengths=lengths,
        sample_rates=torch.full_like(lengths, sample_rate),
    )
