"""Duration-independent SSL trajectory descriptors and bounded alignment."""

import math

import torch
import torch.nn.functional as F


def resample_trajectory(values, points=16):
    """Resample in relative time, preserving embedding location and ordering.

    Each interpolated point is L2-normalized for cosine comparison. No duration,
    timestamps, segment centering, or external annotations enter the descriptor.
    """
    if not isinstance(points, int) or points < 2:
        raise ValueError("points must be an integer >= 2")
    if values.ndim != 2 or len(values) < 2 or not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError("Expected finite floating-point [T,D] trajectory, T >= 2")
    result = F.interpolate(values.T[None].float(), size=points, mode="linear", align_corners=True)[0].T
    if torch.any(torch.linalg.vector_norm(result, dim=-1) <= 1e-8):
        raise ValueError("Cosine descriptors require nonzero interpolated frame vectors")
    return F.normalize(result, dim=-1)


def trajectory_dtw(first, second, band_fraction=0.25):
    """Refine normalized fixed-length trajectories with a constrained DTW path.

    Choose the minimum accumulated cosine-cost path, then report its cost per
    matched step. This is not a minimum-average-cost objective. Alignment uses
    retained resampled trajectories, not the discarded native-frame sequences.
    """
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("DTW inputs must share shape [points,features]")
    if not math.isfinite(band_fraction) or not 0 <= band_fraction <= 1:
        raise ValueError("band_fraction must be in [0,1]")
    costs = (1 - first @ second.T).clamp(0, 2).cpu().tolist()
    size = len(costs)
    band = math.ceil((size - 1) * band_fraction)
    previous = [(math.inf, 0, 0)] * (size + 1)
    previous[0] = (0.0, 0, 0)
    for i in range(1, size + 1):
        current = [(math.inf, 0, 0)] * (size + 1)
        for j in range(max(1, i - band), min(size, i + band) + 1):
            # Stable minimum prefers diagonal steps on ties.
            options = [
                previous[j - 1],
                (previous[j][0], previous[j][1], previous[j][2] + 1),
                (current[j - 1][0], current[j - 1][1], current[j - 1][2] + 1),
            ]
            cost, steps, warps = min(options, key=lambda state: state[0])
            current[j] = (cost + costs[i - 1][j - 1], steps + 1, warps)
        previous = current
    cost, steps, warps = previous[-1]
    return dict(distance=cost / steps, non_diagonal_step_fraction=warps / steps)


@torch.inference_mode()
def nearest_trajectories(
    bank, metadata, query_indices, *, top_k=5, batch_size=32, device="cpu", different_speaker=False
):
    """Exact cosine trajectory search, excluding the query's whole utterance.

    Inputs must be pointwise-normalized [segments,points,features] descriptors.
    Distances average pointwise cosine distance; duration is never consulted.
    Only [query_batch,segments] score blocks are materialized.
    """
    if len(bank) != len(metadata) or bank.ndim != 3:
        raise ValueError("Expected one metadata record per [points,features] descriptor")
    if top_k < 1 or batch_size < 1:
        raise ValueError("top_k and batch_size must be positive")
    vectors = bank.to(device).flatten(1) / math.sqrt(bank.shape[1])
    output = []
    for start in range(0, len(query_indices), batch_size):
        indices = query_indices[start : start + batch_size]
        distances = (1 - vectors[indices] @ vectors.T).clamp(0, 2)
        for offset, index in enumerate(indices):
            query = metadata[index]
            valid = [
                i
                for i, m in enumerate(metadata)
                if (m["split"], m["utt_id"]) != (query["split"], query["utt_id"])
                and (not different_speaker or m["speaker_id"] != query["speaker_id"])
            ]
            if not valid:
                output.append([])
                continue
            candidates = torch.tensor(valid, device=vectors.device)
            values, order = distances[offset, candidates].topk(min(top_k, len(valid)), largest=False, sorted=True)
            output.append(
                [
                    (valid[j], float(distance))
                    for j, distance in zip(order.cpu().tolist(), values.cpu().tolist(), strict=True)
                ]
            )
    return output
