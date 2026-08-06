from __future__ import annotations

from quick_convert.pipelines.evaluation.metrics.base import Metric

#TODO: Implement UAR metric for SER evaluation
class UAR(Metric):
    key = "ser_label"
    ref_key = "ref_ser_label"
    pred_key = "pred_ser_label"

    def __init__(self, device: str = "cpu"):
        self.device = device

    def get_references(self, batch):
        return

    def compute(self, references: list, hypotheses: list) -> dict[str, float]:
        uar = None
        return {"uar": uar}