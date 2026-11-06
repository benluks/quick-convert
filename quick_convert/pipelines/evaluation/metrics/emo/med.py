from __future__ import annotations

import numpy as np

from quick_convert.pipelines.evaluation.metrics.base import Metric


class MeanEuclideanDistanceMetric(Metric):
    """Score paired emotion vectors for one utterance or a whole corpus."""

    key = "ser_embedding"
    ref_key = "ref_ser_embedding"
    pred_key = "pred_ser_embedding"

    def get_references(self, batch):
        return batch.resources[self.key]

    def compute(self, references, hypotheses) -> dict[str, float]:
        refs = np.asarray(references, dtype=float)
        preds = np.asarray(hypotheses, dtype=float)
        if refs.shape != preds.shape or refs.ndim not in (1, 2):
            raise ValueError("References and hypotheses must contain matching emotion vectors.")
        if refs.shape == (0,):
            return {"ser_med": 0.0}
        if refs.shape[-1] == 0 or not np.isfinite(refs).all() or not np.isfinite(preds).all():
            raise ValueError("Emotion vectors must be nonempty and finite.")
        distances = np.linalg.norm(refs - preds, axis=-1)
        return {"ser_med": float(np.mean(distances)) if distances.size else 0.0}
