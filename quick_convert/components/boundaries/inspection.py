"""Pair selection and SSL-only distance diagnostics for trajectory inspection."""

import math

import torch
import torch.nn.functional as F

from .trajectories import resample_trajectory


def segment_key(segment):
    return (segment["split"], segment["utt_id"], segment["path"], segment["start_frame"], segment["end_frame"])


def select_pairs(
    records, *, population="detected", condition="nearest", score_bins=3, pairs_per_stratum=2, duration_ratio_split=1.5
):
    """Deterministic score-rank strata crossed with duration-ratio buckets.

    The display sample deliberately covers the score range and is not a random
    or representative population estimate. Repeated directed pairs are removed.
    """
    if score_bins < 1 or pairs_per_stratum < 1 or not math.isfinite(duration_ratio_split) or duration_ratio_split <= 1:
        raise ValueError("Invalid inspection strata")
    unique = {}
    random_partners = {}
    for record in records:
        if record["population"] != population:
            continue
        if record["condition"] == "random_partner":
            random_partners[segment_key(record["query"])] = record
        if record["condition"] == condition:
            score = record["cosine_distance"]
            ratio = record["duration_ratio"]
            if not math.isfinite(score) or not 0 <= score <= 2 or not math.isfinite(ratio) or ratio < 1:
                raise ValueError("Invalid saved score or duration ratio")
            unique.setdefault((segment_key(record["query"]), segment_key(record["candidate"])), record)
    ranked = sorted(
        unique.values(),
        key=lambda record: (record["cosine_distance"], segment_key(record["query"]), segment_key(record["candidate"])),
    )
    if not ranked:
        raise ValueError(f"No saved matches for {population}/{condition}")
    selected = []
    for index in range(score_bins):
        pool = ranked[index * len(ranked) // score_bins : (index + 1) * len(ranked) // score_bins]
        for stretched in (False, True):
            bucket = [r for r in pool if (r["duration_ratio"] >= duration_ratio_split) == stretched]
            # Evenly spaced ranks include a range within each stratum.
            count = min(pairs_per_stratum, len(bucket))
            indices = torch.linspace(0, len(bucket) - 1, count).round().long().tolist() if count else []
            for offset in indices:
                record = bucket[offset]
                selected.append(
                    dict(
                        record=record,
                        score_stratum=index + 1,
                        duration_bucket="larger_difference" if stretched else "smaller_difference",
                        random_partner=random_partners.get(segment_key(record["query"])),
                    )
                )
    return selected


def compare_trajectories(query, candidate, *, points=16, max_display_frames=128, stretch_factors=(1.5, 2.0)):
    """Real-pair native/grid distances and numerical stretched-copy references.

    Artificial copies stretch already-extracted embeddings, not audio. They
    check interpolation behavior and cannot calibrate natural content identity.
    Native heatmaps may be display-subsampled; scores always use the full
    selected relative-time grid.
    """
    if not isinstance(max_display_frames, int) or max_display_frames < 2:
        raise ValueError("max_display_frames must be >= 2")
    first, second = resample_trajectory(query, points), resample_trajectory(candidate, points)
    normalized = (1 - first @ second.T).clamp(0, 2)
    pointwise = normalized.diagonal()

    def display(values):
        indices = torch.linspace(0, len(values) - 1, min(len(values), max_display_frames)).round().long()
        return F.normalize(values[indices].float(), dim=-1), indices.tolist()

    q, q_indices = display(query)
    c, c_indices = display(candidate)
    references = []
    for factor in stretch_factors:
        if not math.isfinite(factor) or factor <= 1:
            raise ValueError("stretch_factors must be finite and greater than 1")
        count = max(2, round(len(query) * factor))
        stretched = F.interpolate(query.T[None].float(), size=count, mode="linear", align_corners=True)[0].T
        reference = resample_trajectory(stretched, points)
        matrix = (1 - first @ reference.T).clamp(0, 2)
        references.append(
            dict(
                requested_factor=factor,
                native_frames=count,
                realized_frame_count_ratio=count / len(query),
                cosine_distance=float(matrix.diagonal().mean()),
                pointwise_distances=matrix.diagonal().tolist(),
                normalized_distance_matrix=matrix.tolist(),
            )
        )
    mean_q = F.normalize(first.mean(0), dim=0)
    mean_c = F.normalize(second.mean(0), dim=0)
    return dict(
        query_native_frames=len(query),
        candidate_native_frames=len(candidate),
        descriptor_points=points,
        cosine_distance=float(pointwise.mean()),
        pointwise_distances=pointwise.tolist(),
        mean_descriptor_cosine_distance=float((1 - (mean_q * mean_c).sum()).clamp(0, 2)),
        native_distance_matrix=(1 - q @ c.T).clamp(0, 2).tolist(),
        native_query_display_indices=q_indices,
        native_candidate_display_indices=c_indices,
        normalized_distance_matrix=normalized.tolist(),
        stretch_references=references,
    )
