from __future__ import annotations

from quick_convert.components.resources.base import BaseResourceExtractor


class F0Extractor(BaseResourceExtractor):
    """Base for fundamental-frequency extractors; backends subclass this."""

    @property
    def feature_name(self) -> str:
        return "f0"
