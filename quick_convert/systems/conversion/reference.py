"""Reference audio preparation shared by speech conversion systems."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from quick_convert.components.decoders.cosyvoice_reference import CosyVoiceReference
from quick_convert.components.decoders.streams import ConditioningStream
from quick_convert.data import AudioBatch
from quick_convert.systems.conversion.targets import ReferenceSpeech


@dataclass(frozen=True)
class ReferenceCondition:
    """Prepared decoder inputs; representations retain their existing contracts."""

    embedding: torch.Tensor | None = None
    reference: CosyVoiceReference | None = None
    streams: dict[str, ConditioningStream] | None = None

    def decoder_kwargs(self, batch_size, *, device, dtype):
        """Copy and expand target inputs without exposing prepared storage."""
        kwargs = {"speaker_embedding": None}
        if self.embedding is not None:
            if self.embedding.ndim != 2 or self.embedding.shape[0] != 1 or not torch.isfinite(self.embedding).all():
                raise ValueError("Reference speaker embedding must be finite with shape [1, D].")
            kwargs["speaker_embedding"] = self.embedding.to(device=device, dtype=dtype).expand(batch_size, -1).clone()
        if self.reference is not None:
            kwargs["reference"] = self.reference.expand(batch_size, device=device, dtype=dtype)
        if self.streams is not None:
            kwargs["streams"] = {
                name: ConditioningStream(
                    stream.values.to(device=device, dtype=dtype).expand(batch_size, -1, -1).clone(),
                    stream.lengths.to(device=device).expand(batch_size).clone(),
                    True,
                )
                for name, stream in self.streams.items()
            }
        return kwargs


def prepare_reference_audio(target: ReferenceSpeech, load_batch: Callable):
    """Assemble active speech or crop the first recording, preserving provenance.

    ``load_batch`` supplies one recording at the system's input rate and device.
    Feature extraction remains the responsibility of the calling system.
    """
    if target.prompt_seconds is None:
        raise ValueError("Reference audio preparation requires prompt_seconds.")
    if target.prompt_builder is not None:
        template = None

        def recordings():
            nonlocal template
            for path in sorted(target.paths):
                template = load_batch(path)
                yield path, template.waveforms[0, : int(template.lengths[0])], int(template.sample_rates[0])

        prompt = target.prompt_builder(recordings(), target.prompt_seconds)
        return replace(
            template,
            waveforms=prompt.waveform.unsqueeze(0),
            lengths=template.lengths.new_tensor([prompt.waveform.numel()]),
            sample_rates=template.sample_rates.new_tensor([prompt.sample_rate]),
        ), prompt
    batch: AudioBatch = load_batch(target.paths[0])
    count = min(int(batch.lengths[0]), int(target.prompt_seconds * int(batch.sample_rates[0])))
    if count <= 0:
        raise ValueError("Reference prompt must contain at least one sample.")
    return replace(batch, waveforms=batch.waveforms[:, :count].clone(), lengths=batch.lengths.new_tensor([count])), None
