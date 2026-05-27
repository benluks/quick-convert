# quick_convert/data/annotations/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol


class SupportsAnnotations(Protocol):
    annotations: dict[str, Any]


class PathFormatter:
    @staticmethod
    def format(sample: Any, template: str) -> Path:
        path = Path(PathFormatter._get_sample_value(sample, "path"))

        values = {
            "path": path,
            "stem": path.stem,
            "name": path.name,
            "suffix": path.suffix,
            "parent": path.parent,
            "parent_name": path.parent.name,
            "parent_stem": path.parent.stem,
            "grandparent": path.parent.parent,
            "grandparent_name": path.parent.parent.name,
            "grandparent_stem": path.parent.parent.stem,
        }

        if hasattr(sample, "metadata") and sample.metadata:
            values.update(sample.metadata)

        return Path(template.format(**values)).resolve()

    @staticmethod
    def _get_sample_value(sample: Any, key: str) -> Any:
        if isinstance(sample, dict):
            return sample[key]

        if hasattr(sample, key):
            return getattr(sample, key)

        if hasattr(sample, "metadata") and key in sample.metadata:
            return sample.metadata[key]

        raise KeyError(f"Could not find key {key!r} on sample, sample dict, or sample.metadata")


class BaseAnnotationProvider(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def __call__(self, sample: Any) -> Any: ...
