"""Pure, frame-rate-aware boundary measurements on SSL representations.

No encoder, dataset or decoder dependency: the caller supplies one unpadded
[T, D] representation and its frame rate. Detection parameters are explicit.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def change_signal(embeddings: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
    """Return [T-1] changes between successive embedding frames."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("Expected embeddings with shape [T, D], T >= 2")
    if not torch.isfinite(embeddings).all():
        raise ValueError("Embeddings must be finite")
    if metric == "l2":
        return torch.linalg.vector_norm(embeddings[1:] - embeddings[:-1], dim=-1)
    if metric == "cosine":
        return (1 - F.cosine_similarity(embeddings[1:], embeddings[:-1], dim=-1, eps=1e-8)).clamp_min(0)
    raise ValueError(f"Unknown change metric: {metric}")


def detect_boundaries(
    changes: torch.Tensor,
    *,
    threshold: float,
    min_distance_frames: int = 1,
) -> torch.Tensor:
    """Find local maxima above threshold, enforcing distance by peak strength.

    Returned indices are frame-boundary positions in [1, T-1], not indices
    into the change signal. Equal-height adjacent peaks prefer the earliest.
    Threshold is absolute; percentile calibration belongs to the experiment.
    """
    if changes.ndim != 1 or not torch.isfinite(changes).all():
        raise ValueError("Expected a finite 1D change signal")
    if min_distance_frames < 1:
        raise ValueError("min_distance_frames must be >= 1")
    if not torch.isfinite(torch.tensor(threshold)):
        raise ValueError("threshold must be finite")
    if changes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=changes.device)
    # Collapse flat runs before peak finding: the first frame of a maximal
    # plateau wins, including plateaus touching either endpoint. A completely
    # flat signal contains no transition peaks.
    starts = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=changes.device),
            torch.nonzero(changes[1:] != changes[:-1], as_tuple=True)[0] + 1,
        )
    )
    heights = changes[starts]
    if len(starts) == 1:
        return torch.empty(0, dtype=torch.long, device=changes.device)
    left = torch.cat((heights.new_tensor([-math.inf]), heights[:-1]))
    right = torch.cat((heights[1:], heights.new_tensor([-math.inf])))
    candidates = starts[(heights >= threshold) & (heights > left) & (heights > right)]
    ranked = sorted(candidates.tolist(), key=lambda i: (-float(changes[i]), i))
    accepted: list[int] = []
    for index in ranked:
        if all(abs(index - previous) >= min_distance_frames for previous in accepted):
            accepted.append(index)
    return torch.tensor(sorted(i + 1 for i in accepted), dtype=torch.long, device=changes.device)


def segment_measurements(
    boundaries: torch.Tensor,
    *,
    num_frames: int,
    frame_hop_seconds: float,
    include_edges: bool = True,
) -> list[dict]:
    """Measure consecutive segments; explicitly flag recording-edge segments."""
    if num_frames < 2 or not math.isfinite(frame_hop_seconds) or frame_hop_seconds <= 0:
        raise ValueError("num_frames must be >= 2 and frame_hop_seconds positive")
    if boundaries.ndim != 1 or boundaries.is_floating_point() or boundaries.dtype == torch.bool:
        raise ValueError("Boundaries must be a 1D integer tensor")
    points = [int(x) for x in boundaries.tolist()]
    if points != sorted(set(points)) or any(x <= 0 or x >= num_frames for x in points):
        raise ValueError("Boundaries must be unique, ordered and strictly internal")
    edges = [0, *points, num_frames]
    records = []
    for i, (start, end) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        is_edge = i == 0 or i == len(edges) - 2
        if not include_edges and is_edge:
            continue
        records.append(
            {
                "start_frame": start,
                "end_frame": end,
                "start_seconds": start * frame_hop_seconds,
                "end_seconds": end * frame_hop_seconds,
                "duration_ms": (end - start) * frame_hop_seconds * 1000,
                "is_edge": is_edge,
            }
        )
    return records
