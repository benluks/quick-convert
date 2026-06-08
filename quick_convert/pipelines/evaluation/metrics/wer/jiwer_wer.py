from __future__ import annotations

import jiwer

from .....data.types import AudioBatch

from .base import WERMetric

_TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


class JiwerWER(WERMetric):
    def __init__(self, key="transcript"):
        super().__init__(key)

    def compute(self, references: list[str], hypotheses: list[str]) -> float:

        return {
            "wer": float(
                jiwer.wer(
                    references,
                    hypotheses,
                    reference_transform=_TRANSFORM,
                    hypothesis_transform=_TRANSFORM,
                )
            )
        }
