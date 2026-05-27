from pathlib import Path
from typing import Any


class BaseAnnotationProvider:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, sample: Any) -> Any:
        raise NotImplementedError


class PathFormatter:
    @staticmethod
    def format(sample: Any, template: str) -> Path:
        return Path(template.format(**{"sample": sample, "path": sample.path})).resolve()

    @staticmethod
    def _get_sample_value(sample: Any, key: str) -> Any:
        value = sample

        for part in key.split("."):
            if isinstance(value, dict):
                value = value[part]
            else:
                value = getattr(value, part)

        return value
