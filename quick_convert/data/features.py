from pathlib import Path
from typing import Callable

import torch


class SidecarFeatureLoader:
    def __init__(
        self,
        root: str | Path,
        pattern: str,
        key: str,
        loader: Callable[[Path], torch.Tensor] | None = None,
    ):
        self.root = Path(root)
        self.pattern = pattern
        self.key = key
        self.loader = loader or torch.load

    def path_for(self, audio_path: Path) -> Path:
        return self.root / self.pattern.format(
            stem=audio_path.stem,
            name=audio_path.name,
            parent=audio_path.parent.name,
        )

    def load(self, audio_path: Path) -> torch.Tensor:
        return self.loader(self.path_for(audio_path))
