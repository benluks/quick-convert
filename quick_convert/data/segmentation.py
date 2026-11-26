"""Dataset-independent segmentation of source audio intervals."""

import math
from dataclasses import replace
from typing import Protocol

from .durations import resolve_durations
from .types import MetadataSample


class SegmentationPolicy(Protocol):
    def build(self, rows: list[MetadataSample]) -> list[MetadataSample]: ...


class FixedDurationSegmentation:
    """Partition source intervals into non-overlapping duration-limited segments."""

    def __init__(self, duration: float, drop_remainder: bool = False, duration_cache: str | None = None):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive.")
        self.duration = duration
        self.drop_remainder = drop_remainder
        self.duration_cache = duration_cache

    def build(self, rows: list[MetadataSample]) -> list[MetadataSample]:
        if any(list(row.resources) for row in rows):
            raise ValueError("Segmentation of existing resources requires explicit resource slicing.")
        lengths = resolve_durations(rows, cache_path=self.duration_cache)
        segments = []
        for row, length in zip(rows, lengths, strict=True):
            start = row.start or 0.0
            count = math.ceil(length / self.duration)
            for index in range(count):
                left = index * self.duration
                right = min((index + 1) * self.duration, length)
                if right - left <= 1e-9:
                    continue
                if self.drop_remainder and right - left < self.duration - 1e-9:
                    continue
                segments.append(
                    replace(
                        row,
                        utt_id=f"{row.utt_id}__seg{index:06d}",
                        start=start + left,
                        end=start + right,
                        parent_utt_id=row.parent_utt_id or row.utt_id,
                    )
                )
        if len({row.utt_id for row in segments}) != len(segments):
            raise ValueError("Segmentation produced duplicate utterance IDs.")
        return segments
