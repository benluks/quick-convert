from __future__ import annotations

from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn.functional as F

from ...metrics.eer import compute_eer
from .trials import ASVTrial


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
    enrollment, test = _normalize_embeddings(enrollment, test)
    return enrollment @ test.T


def _normalize_embeddings(enrollment, test):
    if enrollment.ndim != 2 or test.ndim != 2 or enrollment.shape[-1] != test.shape[-1]:
        raise ValueError("Enrollment and test embeddings must be matrices with matching feature dimensions.")
    for values in (enrollment, test):
        if not torch.isfinite(values).all() or torch.any(values.norm(dim=-1) == 0):
            raise ValueError("Scoring requires finite, nonzero embeddings, including enrollment averages.")
    return F.normalize(enrollment, dim=-1), F.normalize(test, dim=-1)


@torch.inference_mode()
def evaluate_trials(
    enrollment: dict[str, torch.Tensor],
    test: dict[str, torch.Tensor],
    trials: Iterable[ASVTrial],
) -> dict[str, float | int]:
    """Score explicit trials from enrollment and test vectors, then report EER.

    Enrollment keys identify speakers and test keys identify utterances. Trial
    labels determine the target class. Only listed pairs are scored.
    """
    if not enrollment or not test:
        raise ValueError("Enrollment and test embeddings must both be nonempty.")
    enroll_index = {key: index for index, key in enumerate(enrollment)}
    test_index = {key: index for index, key in enumerate(test)}
    enroll_matrix, test_matrix = _normalize_embeddings(
        torch.stack(list(enrollment.values())), torch.stack(list(test.values()))
    )

    target_scores = []
    non_target_scores = []
    trial_iter = iter(trials)
    while batch := list(islice(trial_iter, 4096)):
        enroll_rows = []
        test_rows = []
        for trial in batch:
            if trial.enroll_id not in enroll_index or trial.test_id not in test_index:
                raise ValueError(f"Unknown enrollment or test ID in trial: {trial}")
            enroll_rows.append(enroll_index[trial.enroll_id])
            test_rows.append(test_index[trial.test_id])
        scores = (enroll_matrix[enroll_rows] * test_matrix[test_rows]).sum(dim=-1).cpu()
        target = torch.tensor([trial.target for trial in batch], dtype=torch.bool)
        if target.any():
            target_scores.append(scores[target])
        if not target.all():
            non_target_scores.append(scores[~target])

    if not target_scores or not non_target_scores:
        raise ValueError("ASV evaluation requires both target and non-target trials.")
    targets = torch.cat(target_scores)
    non_targets = torch.cat(non_target_scores)
    eer, threshold = compute_eer(targets, non_targets)
    return {
        "eer": eer,
        "eer_threshold": threshold,
        "num_target_trials": len(targets),
        "num_non_target_trials": len(non_targets),
    }
