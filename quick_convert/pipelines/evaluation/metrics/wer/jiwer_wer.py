from __future__ import annotations

from ..utils import load_lines
from .base import WERMetric


class JiwerWER(WERMetric):
    def __init__(self, key="transcript"):
        super().__init__(key)

        import jiwer

        self._TRANSFORM = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.Strip(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )

    def compute(self, references: list[str], hypotheses: list[str]) -> float:
        references = load_lines(references)
        hypotheses = load_lines(hypotheses)
        return {
            "wer": float(
                jiwer.wer(
                    references,
                    hypotheses,
                    reference_transform=self._TRANSFORM,
                    hypothesis_transform=self._TRANSFORM,
                )
            )
        }
