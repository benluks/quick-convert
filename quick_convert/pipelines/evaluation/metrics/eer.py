from __future__ import annotations

import torch


def compute_eer(
    target_scores: torch.Tensor,
    non_target_scores: torch.Tensor,
) -> tuple[float, float]:
    """Return discrete EER and threshold for acceptance at score >= threshold.

    Select the observed threshold minimizing |FAR - FRR| and average those
    rates. Ties select the lowest threshold; no interpolation is performed.
    """
    target_scores = target_scores.detach().to(device="cpu", dtype=torch.float64)
    non_target_scores = non_target_scores.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(target_scores).all() or not torch.isfinite(non_target_scores).all():
        raise ValueError("ASV scores must be finite.")
    if target_scores.numel() == 0:
        raise ValueError("target_scores must not be empty.")

    if non_target_scores.numel() == 0:
        raise ValueError("non_target_scores must not be empty.")

    target_scores = torch.sort(target_scores.flatten()).values

    non_target_scores = torch.sort(non_target_scores.flatten()).values

    thresholds = torch.unique(
        torch.cat(
            [
                target_scores,
                non_target_scores,
            ]
        )
    )

    false_reject = (
        torch.searchsorted(
            target_scores,
            thresholds,
            right=False,
        ).float()
        / target_scores.numel()
    )

    false_accept = (
        non_target_scores.numel()
        - torch.searchsorted(
            non_target_scores,
            thresholds,
            right=False,
        )
    ).float() / non_target_scores.numel()

    index = torch.argmin(torch.abs(false_reject - false_accept))

    eer = (false_reject[index] + false_accept[index]) / 2

    return float(eer), float(thresholds[index])
