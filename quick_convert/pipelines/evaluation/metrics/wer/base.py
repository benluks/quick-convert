from __future__ import annotations

from abc import abstractmethod
from .. import Metric


class WERMetric(Metric):
    ref_key = "ref_transcript"
    pred_key = "hyp_transcript"

    def __init__(self, key="transcript"):
        self.key = key

    @abstractmethod
    def compute(self, references: list[str], hypotheses: list[str]) -> float: ...

    def get_references(self, batch):
        return batch.resources[self.key]
