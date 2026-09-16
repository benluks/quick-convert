from __future__ import annotations

from .base import WERMetric


class JiwerWER(WERMetric):
    def __init__(self, key="transcript"):
        super().__init__(key)

        try:
            import jiwer
        except ImportError as error:
            raise ImportError("WER evaluation requires the `asr` extra.") from error

        self._jiwer = jiwer

        self._TRANSFORM = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.Strip(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )

    def compute(self, references: str | list[str], hypotheses: str | list[str]) -> dict[str, float]:
        if isinstance(references, str):
            references = [references]
        if isinstance(hypotheses, str):
            hypotheses = [hypotheses]
        return {
            "wer": float(
                self._jiwer.wer(
                    references,
                    hypotheses,
                    reference_transform=self._TRANSFORM,
                    hypothesis_transform=self._TRANSFORM,
                )
            )
        }
