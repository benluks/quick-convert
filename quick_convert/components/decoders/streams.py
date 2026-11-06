"""Named frame-wise conditions and independent context masking for mel generation."""

import json
import math
from dataclasses import dataclass

import torch
from torch import nn

from quick_convert.utils.masking import make_padding_mask


@dataclass(frozen=True)
class ConditioningStream:
    """Values [B, T, D] and valid lengths [B], on the mel frame clock.

    Training values cover the source. Inference values contain only a reference
    prefix; different streams may have different lengths or be absent. Set
    ``is_prefix=True`` for training prefixes encoded after waveform cropping;
    these are not sampled or shortened again by the decoder.
    """

    values: torch.Tensor
    lengths: torch.Tensor
    is_prefix: bool = False

    def validate(self, batch_size, dim, device, dtype):
        if self.values.ndim != 3 or self.values.shape[0] != batch_size or self.values.shape[2] != dim:
            raise ValueError("Stream values must have shape [batch, time, declared dimension].")
        if self.lengths.shape != (batch_size,) or self.lengths.dtype != torch.long:
            raise ValueError("Stream lengths must be int64 vectors with one entry per sample.")
        if self.values.device != device or self.lengths.device != device or self.values.dtype != dtype:
            raise ValueError("Streams must share the content device and dtype.")
        if torch.any(self.lengths < 0) or torch.any(self.lengths > self.values.shape[1]):
            raise ValueError("Stream lengths exceed storage or are negative.")
        valid = make_padding_mask(self.lengths, max_length=self.values.shape[1])
        if not torch.isfinite(self.values[valid]).all():
            raise ValueError("Valid stream values must be finite.")


class StreamConditioning(nn.Module):
    """Concatenate fixed named streams and one availability channel per stream.

    ``mel`` is reserved for automatically extracted training target mels. Other
    streams are supplied explicitly. No resampling or rate inference is done.
    Stream order follows ``dimensions`` and is part of the checkpoint contract.
    """

    def __init__(
        self,
        dimensions: dict[str, int],
        prefix_fraction: tuple[float, float] = (0.0, 0.3),
        context_mask_fraction: tuple[float, float] = (0.0, 0.5),
        stream_drop_probability: float = 0.2,
    ):
        super().__init__()
        self.dimensions = dict(dimensions)
        if not self.dimensions or any(
            not isinstance(name, str) or not name or type(dim) is not int or dim <= 0
            for name, dim in self.dimensions.items()
        ):
            raise ValueError("Stream dimensions must be a nonempty mapping of names to positive integers.")
        for interval in (prefix_fraction, context_mask_fraction):
            if len(interval) != 2 or not 0 <= interval[0] <= interval[1] <= 1:
                raise ValueError("Mask fractions must be ordered intervals in [0, 1].")
        if not math.isfinite(stream_drop_probability) or not 0 <= stream_drop_probability <= 1:
            raise ValueError("Mask probabilities must be finite and in [0, 1].")
        # Equal-width stream permutations must not silently reinterpret saved weights.
        layout = json.dumps(list(self.dimensions.items())).encode()
        self.register_buffer("_layout", torch.tensor(list(layout), dtype=torch.uint8))
        self.prefix_fraction = tuple(prefix_fraction)
        self.context_mask_fraction = tuple(context_mask_fraction)
        self.stream_drop_probability = stream_drop_probability

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        saved = state_dict.get(prefix + "_layout")
        if saved is not None and not torch.equal(saved.cpu(), self._layout.cpu()):
            raise ValueError("Checkpoint stream names, order, or dimensions differ from this model.")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def output_dim(self):
        """Packed condition width, including availability indicators."""
        return sum(self.dimensions.values()) + len(self.dimensions)

    def _validate(self, streams, template):
        unknown = streams.keys() - self.dimensions.keys()
        if unknown:
            raise ValueError(f"Undeclared conditioning streams: {sorted(unknown)}")
        for name, stream in streams.items():
            stream.validate(template.shape[0], self.dimensions[name], template.device, template.dtype)

    def sample_prefix_lengths(self, lengths):
        """Choose prefix lengths before a contextual encoder reads any waveform."""
        if torch.any(lengths <= 0):
            raise ValueError("Target lengths must be positive.")
        fractions = torch.empty(lengths.shape, device=lengths.device).uniform_(*self.prefix_fraction)
        prefix_lengths = torch.minimum((lengths * fractions).long(), lengths - 1)
        keep = torch.rand(lengths.shape, device=lengths.device) >= self.stream_drop_probability
        return prefix_lengths * keep

    def training_conditions(self, streams, lengths, template, *, prefix_lengths=None):
        """Sample independent prefixes and score only their shared continuation."""
        self._validate(streams, template)
        if torch.any(lengths <= 0) or torch.any(lengths > template.shape[1]):
            raise ValueError("Target lengths must be positive and fit the mel storage.")
        prefixes = {}
        for name, stream in streams.items():
            if stream.is_prefix:
                if prefix_lengths is None and torch.any(stream.lengths >= lengths):
                    raise ValueError("A supplied training prefix must leave at least one continuation frame.")
                prefixes[name] = stream
            else:
                if torch.any(stream.lengths != lengths):
                    raise ValueError(f"Training stream {name!r} must cover each target on the mel clock.")
                prefixes[name] = ConditioningStream(stream.values, self.sample_prefix_lengths(lengths), is_prefix=True)
        align_ends = prefix_lengths is None
        packed, prefix_lengths = self.prefix_conditions(prefixes, template, lengths=prefix_lengths)
        # Independently hide bricks inside each supplied prefix without exposing
        # auxiliary context in the continuation.
        packed = packed.clone()
        offset = 0
        for name, dim in self.dimensions.items():
            if name in prefixes:
                for i, size in enumerate(prefixes[name].lengths.tolist()):
                    size = min(size, int(prefix_lengths[i]))
                    fraction = torch.empty((), device=lengths.device).uniform_(*self.context_mask_fraction).item()
                    count = min(size, math.ceil(size * fraction))
                    start = int(prefix_lengths[i]) - size if align_ends else 0
                    start += int(torch.randint(size - count + 1, (), device=lengths.device))
                    packed[i, start : start + count, offset : offset + dim + 1] = 0
            offset += dim + 1
        packed = torch.nn.functional.pad(packed, (0, 0, 0, template.shape[1] - packed.shape[1]))
        positions = torch.arange(template.shape[1], device=lengths.device)[None]
        target = (positions >= prefix_lengths[:, None]) & (positions < lengths[:, None])
        return packed, target

    def prefix_conditions(self, streams, template, *, lengths=None):
        """Pack streams on a supplied content clock, or align their ends by default."""
        self._validate(streams, template)
        align_ends = lengths is None
        if align_ends:
            lengths = torch.zeros(template.shape[0], device=template.device, dtype=torch.long)
            for stream in streams.values():
                lengths = torch.maximum(lengths, stream.lengths)
        time = int(lengths.max())
        packed = []
        for name, dim in self.dimensions.items():
            values = template.new_zeros((template.shape[0], time, dim))
            available = torch.zeros((template.shape[0], time), device=template.device, dtype=torch.bool)
            if name in streams:
                stream = streams[name]
                for i, size in enumerate(stream.lengths.tolist()):
                    end = int(lengths[i])
                    size = min(size, end)
                    start = end - size if align_ends else 0
                    values[i, start : start + size] = stream.values[i, :size]
                    available[i, start : start + size] = True
            packed.extend((values, available[..., None].to(template.dtype)))
        return torch.cat(packed, dim=-1), lengths
