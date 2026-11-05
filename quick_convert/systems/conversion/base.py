"""Structural contract for systems used by dataset conversion workflows."""

from typing import Protocol

import torch
from torch.nn.utils.rnn import pad_sequence

from quick_convert.data import AudioBatch, GeneratedAudio
from quick_convert.systems.conversion.targets import PreparedTarget, ReferenceSpeech, SourceVoice, SpeakerIdentity


class SpeechConversionSystem(Protocol):
    """Prepare backend-specific voice conditioning and convert source batches.

    Target selection is a workflow concern. Systems validate the representation
    and model provenance of prepared conditioning before applying it.
    """

    def prepare_target(
        self, target: ReferenceSpeech | SpeakerIdentity | SourceVoice | PreparedTarget
    ) -> PreparedTarget: ...

    def convert_batch(self, batch: AudioBatch, *, target: PreparedTarget) -> GeneratedAudio: ...


def generate_audio_batch(batch: AudioBatch, operation, *, sample_rate: int, **kwargs) -> GeneratedAudio:
    """Apply a single-item operation and preserve exact output lengths.

    This default adapter is intentionally sequential. Implementations with
    native backend batching may override :meth:`convert_batch` while
    preserving the same output contract.
    """
    waveforms = []

    for sample in batch:
        if sample.waveform is None:
            waveform = operation(sample.path, **kwargs)
        else:
            if sample.sample_rate is None:
                raise ValueError(f"Loaded sample {sample.utt_id!r} has no sample rate.")
            waveform = operation(
                sample.waveform,
                sample_rate=int(sample.sample_rate),
                **kwargs,
            )

        if waveform.ndim == 2 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        elif waveform.ndim != 1:
            raise ValueError(
                "Conversion systems must generate mono waveforms with shape "
                f"(time,) or (1, time); got {tuple(waveform.shape)} for sample {sample.utt_id!r}."
            )
        waveforms.append(waveform)

    if not waveforms:
        raise ValueError("Cannot convert an empty batch.")

    lengths = torch.tensor(
        [waveform.shape[-1] for waveform in waveforms],
        dtype=torch.long,
        device=waveforms[0].device,
    )
    return GeneratedAudio(
        waveforms=pad_sequence(waveforms, batch_first=True),
        lengths=lengths,
        sample_rate=sample_rate,
    )
