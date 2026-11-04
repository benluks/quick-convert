"""Deterministic speech selection and smooth concatenation for reference prompts."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EnergyActivityDetector:
    """Return active sample intervals using non-overlapping RMS frames.

    The threshold is absolute dBFS for normalized floating-point audio. This
    removes quiet regions, not background noise above the threshold. A future
    VAD can implement the same ``(waveform, sample_rate) -> intervals`` call.
    Detection runs on CPU in float64 for repeatable frame decisions.
    """

    threshold_db: float = -40.0
    frame_seconds: float = 0.02
    padding_seconds: float = 0.02

    def __post_init__(self):
        if not math.isfinite(self.threshold_db) or self.threshold_db >= 0:
            raise ValueError("threshold_db must be finite and negative.")
        if not math.isfinite(self.frame_seconds) or self.frame_seconds <= 0:
            raise ValueError("frame_seconds must be finite and positive.")
        if not math.isfinite(self.padding_seconds) or self.padding_seconds < 0:
            raise ValueError("padding_seconds must be finite and nonnegative.")

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> tuple[tuple[int, int], ...]:
        """Return ordered, disjoint half-open intervals in source samples."""
        if sample_rate <= 0 or waveform.ndim != 1 or not waveform.is_floating_point():
            raise ValueError("Activity detection requires mono floating-point audio and a positive sample rate.")
        if not torch.isfinite(waveform).all():
            raise ValueError("Activity detection requires finite audio.")
        if waveform.numel() == 0:
            return ()
        frame = max(1, round(self.frame_seconds * sample_rate))
        values = waveform.detach().to(device="cpu", dtype=torch.float64)
        padding = (-values.numel()) % frame
        frames = F.pad(values, (0, padding)).reshape(-1, frame)
        counts = torch.full((len(frames),), frame, dtype=torch.float64)
        counts[-1] -= padding
        active = frames.square().sum(-1) / counts > 10 ** (self.threshold_db / 10)
        edges = torch.diff(F.pad(active.to(torch.int8), (1, 1)))
        margin = round(self.padding_seconds * sample_rate)
        intervals = []
        for start, end in zip(torch.where(edges == 1)[0], torch.where(edges == -1)[0], strict=True):
            start = max(0, int(start) * frame - margin)
            end = min(values.numel(), int(end) * frame + margin)
            if intervals and start <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], end)
            else:
                intervals.append((start, end))
        return tuple(intervals)


@dataclass(frozen=True)
class PromptSegment:
    """One source interval and the overlap with its preceding prompt segment."""

    path: Path
    start_sample: int
    end_sample: int
    overlap_samples: int = 0


@dataclass(frozen=True)
class AudioPrompt:
    """Assembled mono waveform and enough source information to reproduce it."""

    waveform: torch.Tensor
    sample_rate: int
    segments: tuple[PromptSegment, ...]


@dataclass(frozen=True)
class SpeechPromptBuilder:
    """Fill a duration from active intervals, with linear crossfades and edge fades.

    Recordings are consumed in caller-supplied order at one common sample rate.
    The duration is measured after overlap; insufficient active audio raises
    instead of repeating speech or silently shortening the prompt.
    """

    detector: Callable[[torch.Tensor, int], Iterable[tuple[int, int]]] = field(default_factory=EnergyActivityDetector)
    crossfade_seconds: float = 0.005

    def __post_init__(self):
        if not callable(self.detector):
            raise TypeError("detector must return sample intervals from (waveform, sample_rate).")
        if not math.isfinite(self.crossfade_seconds) or self.crossfade_seconds < 0:
            raise ValueError("crossfade_seconds must be finite and nonnegative.")

    def description(self) -> dict:
        """Describe the detector and fade settings for conversion provenance."""
        from dataclasses import asdict, is_dataclass

        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "crossfade_seconds": self.crossfade_seconds,
            "detector": {
                "type": f"{type(self.detector).__module__}.{type(self.detector).__qualname__}",
                "parameters": asdict(self.detector) if is_dataclass(self.detector) else {},
            },
        }

    def __call__(self, recordings: Iterable[tuple[Path, torch.Tensor, int]], seconds: float) -> AudioPrompt:
        """Build a prompt without changing caller-owned audio or random state."""
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Prompt duration must be finite and positive.")
        output = None
        segments = []
        rate = None
        for path, waveform, sample_rate in recordings:
            if (
                not isinstance(sample_rate, int)
                or sample_rate <= 0
                or waveform.ndim != 1
                or not waveform.is_floating_point()
                or not torch.isfinite(waveform).all()
            ):
                raise ValueError("Prompt recordings require finite mono audio and positive integer sample rates.")
            if rate is None:
                rate = sample_rate
                count = round(seconds * rate)
                fade = round(self.crossfade_seconds * rate)
                if count < 1:
                    raise ValueError("Prompt duration must contain at least one sample.")
            if sample_rate != rate:
                raise ValueError("Prompt recordings must use one common sample rate.")
            previous_end = 0
            for start, end in self.detector(waveform, sample_rate):
                if not (
                    isinstance(start, int) and isinstance(end, int) and previous_end <= start < end <= waveform.numel()
                ):
                    raise ValueError("Activity detector returned invalid or overlapping sample intervals.")
                previous_end = end
                size = 0 if output is None else output.numel()
                overlap = min(fade, size, end - start - 1)
                end = min(end, start + count - size + overlap)
                chunk = waveform[start:end].clone()
                if output is None:
                    output = chunk
                elif overlap:
                    weights = torch.linspace(0, 1, overlap + 2, device=output.device, dtype=output.dtype)[1:-1]
                    joined = output[-overlap:] * (1 - weights) + chunk[:overlap] * weights
                    output = torch.cat((output[:-overlap], joined, chunk[overlap:]))
                else:
                    output = torch.cat((output, chunk))
                segments.append(PromptSegment(Path(path), start, end, overlap))
                if output.numel() == count:
                    edge = min(fade, count // 2)
                    if edge:
                        ramp = torch.linspace(0, 1, edge, device=output.device, dtype=output.dtype)
                        output[:edge] *= ramp
                        output[-edge:] *= ramp.flip(0)
                    return AudioPrompt(output, rate, tuple(segments))
        available = 0 if output is None else output.numel() / rate
        raise ValueError(f"Not enough active reference audio: requested {seconds:g}s, retained {available:.3f}s.")
