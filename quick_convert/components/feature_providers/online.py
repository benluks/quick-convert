from __future__ import annotations

from typing import Any, Optional

from quick_convert.data.types import AudioBatch, AudioSample


class OnlineFeatureProvider:
    def __init__(
        self,
        extractor: Any,
        key: Optional[str] = None,
    ):
        self.extractor = extractor
        self.key = self.extractor.feature_name if key is None else key

    def provide_sample(self, sample: AudioSample) -> dict[str, Any]:
        return self.extractor.extract_sample(sample)

    def provide_batch(self, batch: AudioBatch) -> dict[str, Any]:
        return self.extractor.extract_batch(batch)
