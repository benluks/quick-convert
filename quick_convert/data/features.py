from __future__ import annotations

from pathlib import Path


class PatternSidecarFeatureResolver:
    def __init__(
        self,
        key: str,
        root: str | Path,
        pattern: str,
        load: bool = False,
        loader=None,
        **format_kwargs,
    ):
        self.key = key
        self.root = Path(root)
        self.pattern = pattern
        self.load = load
        self.loader = loader
        self.format_kwargs = format_kwargs

    def resolve(self, sample):
        path = self.root / self.pattern.format(
            stem=sample.path.stem,
            name=sample.path.name,
            split=sample.split,
            spk_id=sample.spk_id,
            **self.format_kwargs,
        )

        if self.load:
            return {self.key: self.loader(path)}

        return {self.key: path}
