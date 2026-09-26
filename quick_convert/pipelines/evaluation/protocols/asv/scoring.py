from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_score(
    enrollment: torch.Tensor,
    test: torch.Tensor,
) -> torch.Tensor:
    return F.cosine_similarity(
        enrollment,
        test,
        dim=-1,
    )


def cosine_score_matrix(
    enrollment: torch.Tensor,
    test: torch.Tensor,
) -> torch.Tensor:
    """Return [enrollment speakers, test utterances] cosine similarities."""
    if enrollment.ndim != 2 or test.ndim != 2 or enrollment.shape[-1] != test.shape[-1]:
        raise ValueError("Enrollment and test embeddings must be matrices with matching feature dimensions.")
    for values in (enrollment, test):
        if not torch.isfinite(values).all() or torch.any(values.norm(dim=-1) == 0):
            raise ValueError("Scoring requires finite, nonzero embeddings, including enrollment averages.")
    enrollment = F.normalize(enrollment, dim=-1)
    test = F.normalize(test, dim=-1)

    return enrollment @ test.T
