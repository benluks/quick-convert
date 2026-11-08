"""Aggregation shared by speaker enrollment and voice target preparation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from quick_convert.data import AudioBatch


@dataclass(frozen=True)
class SpeakerEmbeddingAggregation:
    """Reduce one embedding per recording without changing its representation.

    ``mean`` gives each recording equal weight. ``duration_weighted`` uses valid
    audio duration in seconds, excluding batch padding. Embeddings are never
    normalized here: cosine scoring or the consuming decoder owns that choice.
    Concatenation happens before embedding extraction and is not an embedding
    reduction strategy.
    """

    method: Literal["mean", "duration_weighted"] = "mean"

    def __post_init__(self) -> None:
        if self.method not in {"mean", "duration_weighted"}:
            raise ValueError("Speaker embedding aggregation must be 'mean' or 'duration_weighted'.")

    def __call__(
        self,
        embeddings: torch.Tensor,
        *,
        durations: Sequence[float] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate ``[recordings, dimension]`` embeddings into one ``[dimension]`` vector."""
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("Speaker embeddings must be a tensor.")
        if embeddings.ndim != 2 or min(embeddings.shape) == 0:
            raise ValueError("Speaker aggregation requires nonempty embeddings with shape [N, D].")
        if not embeddings.is_floating_point() or not torch.isfinite(embeddings).all():
            raise ValueError("Speaker embeddings must contain finite floating-point values.")
        weights = None
        if durations is not None:
            weights = torch.as_tensor(durations, dtype=torch.float64, device="cpu")
            if weights.shape != (embeddings.shape[0],):
                raise ValueError("Expected one duration per recording.")
            if not torch.isfinite(weights).all() or torch.any(weights <= 0):
                raise ValueError("Recording durations must be finite and positive.")
        if self.method == "mean":
            return embeddings.mean(dim=0)
        if weights is None:
            raise ValueError("Duration-weighted aggregation requires recording durations.")
        weights = weights / weights.max()
        weights = weights / weights.sum()
        # Accumulate half-precision embeddings in float32, preserving the public
        # embedding dtype and avoiding duration overflow in half precision.
        work = embeddings.float() if embeddings.dtype in {torch.float16, torch.bfloat16} else embeddings
        weights = weights.to(device=work.device, dtype=work.dtype)
        return (work * weights[:, None]).sum(dim=0).to(embeddings.dtype)


def audio_durations(batch: AudioBatch) -> tuple[float, ...]:
    """Return valid recording durations from audio lengths and per-item rates.

    Waveform padding contributes no duration. Rates may differ between items;
    callers choose their own resampling/extraction stage. This function reads
    metadata without changing the batch or requiring a speaker backend.
    """
    lengths, rates = batch.lengths, batch.sample_rates
    count = len(batch)
    if lengths is None or rates is None or lengths.shape != (count,) or rates.shape != (count,):
        raise ValueError("Recording durations require one audio length and sample rate per batch item.")
    integer_dtypes = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
    if lengths.dtype not in integer_dtypes or rates.dtype not in integer_dtypes:
        raise ValueError("Audio lengths and sample rates must use integer values.")
    if torch.any(lengths <= 0) or torch.any(rates <= 0):
        raise ValueError("Audio lengths and sample rates must be positive.")
    if batch.waveforms is not None and torch.any(lengths > batch.waveforms.shape[-1]):
        raise ValueError("An audio length exceeds the padded waveform size.")
    durations = lengths.detach().to(device="cpu", dtype=torch.float64) / rates.detach().to(
        device="cpu", dtype=torch.float64
    )
    return tuple(durations.tolist())
