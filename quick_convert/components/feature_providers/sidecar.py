from __future__ import annotations

from pathlib import Path
from typing import Any

from ...data.types import AudioBatch, AudioSample


class PatternSidecarFeatureProvider:
    def __init__(self, key: str, root, pattern: str, load: bool = False, loader=None, **format_kwargs):
        self.key = key
        self.root = Path(root)
        self.pattern = pattern
        self.load = load
        self.loader = loader
        self.format_kwargs = format_kwargs

    def resolve_path(self, sample: AudioSample) -> Path:
        return self.root / self.pattern.format(
            stem=sample.path.stem,
            name=sample.path.name,
            split=sample.split,
            spk_id=sample.spk_id,
            **self.format_kwargs,
        )

    def provide_value(self, sample: AudioSample) -> Any:
        path = self.resolve_path(sample)
        return self.loader(path) if self.load else path

    def provide_sample(self, sample: AudioSample) -> dict[str, Any]:
        return {self.key: self.provide_value(sample)}

    def provide_batch(self, batch: AudioBatch) -> dict[str, list[Any]]:
        return {self.key: [self.provide_value(sample) for sample in batch]}
