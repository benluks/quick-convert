"""Reference-pool conditioning for SSL reconstruction."""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from quick_convert.components.conditioning.nearest_neighbour import NearestNeighbourMatching
from quick_convert.systems.conversion.targets import PreparedTarget, ReferenceSpeech
from quick_convert.utils.audio import load_audio_input


class NearestNeighbourReferenceConditioning(nn.Module):
    """Prepare reference frames once and match source frames before decoding."""

    representation = "ssl-nearest-neighbour-v1"

    def __init__(self, matcher: nn.Module | None = None):
        super().__init__()
        self.matcher = matcher if matcher is not None else NearestNeighbourMatching()

    @staticmethod
    def _encode(system, batch):
        content = system.get_resource(batch, "content")
        if content.lengths is None:
            raise ValueError("Nearest-neighbour conditioning requires content lengths.")
        features, lengths, _ = system.encode_features(content.values, content.lengths, metadata=content.metadata)
        return features, lengths

    def _pool(self, system, target):
        if target.space != system.target_space or target.representation != self.representation:
            raise ValueError("Prepared target belongs to an incompatible model or representation.")
        pool = target.conditioning
        if not isinstance(pool, torch.Tensor) or pool.ndim != 2 or min(pool.shape) == 0:
            raise ValueError("Expected a nonempty reference frame pool.")
        if not pool.is_floating_point() or not torch.isfinite(pool).all():
            raise ValueError("Reference pool must contain finite floating-point features.")
        return pool

    @torch.inference_mode()
    def prepare_target(self, system, target):
        if system.training:
            raise RuntimeError("Call eval() before preparing targets.")
        if isinstance(target, PreparedTarget):
            self._pool(system, target)
            return target
        if not isinstance(target, ReferenceSpeech):
            raise TypeError("Expected ReferenceSpeech or PreparedTarget.")
        if target.aggregation is not None or target.prompt_seconds is not None or target.allow_missing_streams:
            raise ValueError("Nearest-neighbour pools do not support embedding aggregation or reference prompts.")
        sample_rate = system.online_encoders["content"].sample_rate
        frames = []
        for path in target.paths:
            waveform = load_audio_input(path, target_sample_rate=sample_rate, mono=True).reshape(-1)
            features, lengths = self._encode(system, system._reference_batch(waveform, sample_rate))
            frames.append(features[0, : int(lengths[0])])
        prepared = PreparedTarget(torch.cat(frames), system.target_space, self.representation, target.paths)
        self._pool(system, prepared)
        return prepared

    @torch.inference_mode()
    def convert_batch(self, system, batch, *, target):
        if system.training:
            raise RuntimeError("Call eval() before converting speech.")
        pool = self._pool(system, target)
        features, lengths = self._encode(system, batch)
        pool = pool.to(device=features.device, dtype=features.dtype)
        matched = pad_sequence(
            [self.matcher(x[: int(n)], pool) for x, n in zip(features, lengths, strict=True)],
            batch_first=True,
        )
        generation = system.decode_features(matched, lengths, system.speaker_condition(batch), run_vocoder=True)
        if generation.audio is None:
            raise RuntimeError("Conversion requires generated audio.")
        return generation.audio

    @staticmethod
    def conditioning_streams(system, batch, target_lengths=None):
        return {}
