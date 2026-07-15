from typing import Any
from quick_convert.data.resources.base import BaseResourceProvider
from pathlib import Path

from quick_convert.utils.paths import SamplePathFormatter

class SpeakerIDProvider(BaseResourceProvider):
    def __init__(
        self,
        name: str = "speaker_id",
        path_template: str | None = None,
        speaker_key: str = "path.parent.parent.name",
    ) -> None:
        super().__init__(name=name)

        self.path_template = path_template
        self.speaker_key = speaker_key

        self._speaker2int: dict[str, int] = {}
        self._cache: dict[Path, int] = {}

    def __call__(self, sample: Any) -> str:
        path = self._resolve_path(sample)

        if path not in self._cache:
            self._cache[path] = self._get_speaker_id(path)
        else:
            return self._cache[path]

    def _get_speaker_id(self, path: Path) -> int:
        speaker_id = str(self._get_sample_value(path, self.speaker_key))
        if speaker_id not in self._speaker2int:
            self._speaker2int[speaker_id] = len(self._speaker2int)
        return self._speaker2int[speaker_id]

    def _resolve_path(self, sample: Any) -> Path:
        if self.path_template is not None:
            return SamplePathFormatter.format(sample, self.path_template)
        else:
            raise ValueError(" path_template must be provided.")

    def _get_sample_value(self, sample: Any, key: str) -> Any:
        return SamplePathFormatter._get_sample_value(sample, key)