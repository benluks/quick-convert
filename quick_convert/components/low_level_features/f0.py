from __future__ import annotations

from ..feature_extractors.base import BaseFeatureExtractor


class F0Extractor(BaseFeatureExtractor):
    """Base for fundamental-frequency extractors; backends subclass this."""

    @property
    def feature_name(self) -> str:
        return "f0"
