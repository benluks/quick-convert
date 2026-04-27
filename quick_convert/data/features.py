from pathlib import Path

from .types import AudioSample


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


def resolve_emotion_compensation_xvector_path(sample: AudioSample, root: Path, step: int) -> Path:
    return root / f"{sample.path.stem}_{step}.xvector"
