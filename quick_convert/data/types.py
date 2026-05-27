from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class MetadataSample:
    path: Path
    split: Optional[str] = None
    spk_id: Optional[str] = None
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedSample(MetadataSample):
    waveform: float["1 t"] | None = None
    sample_rate: int | None = None


@dataclass(frozen=True)
class AudioSample(MetadataSample):
    waveform: Optional[float["1 t"]] = None
    sample_rate: Optional[int] = None


@dataclass(frozen=True)
class MetadataBatch:
    paths: list[Path]
    splits: list[str | None]
    spk_ids: list[str | None]
    annotations: dict[str, Any]


@dataclass(frozen=True)
class LoadedBatch(MetadataBatch):
    waveforms: float["b t"]
    lengths: int["b"]
    sample_rates: int["b"]


@dataclass(frozen=True)
class AudioBatch(MetadataBatch):
    waveforms: Optional[float["b t"]] = None
    lengths: Optional[int["b"]] = None
    sample_rates: Optional[int["b"]] = None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> AudioSample:
        return AudioSample(
            path=self.paths[idx],
            split=self.splits[idx],
            spk_id=self.spk_ids[idx],
            waveform=self.waveforms[idx] if self.waveforms is not None else None,
            sample_rate=self.sample_rates[idx] if self.sample_rates is not None else None,
            # features={key: value[idx] for key, value in self.features.items()},
            annotations={key: value[idx] for key, value in self.annotations.items()},
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
