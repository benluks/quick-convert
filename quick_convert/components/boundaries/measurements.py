"""Measurements on one unpadded embedding trajectory and an explicit time grid."""

import math

import torch
import torch.nn.functional as F

from .detection import change_signal, detect_boundaries, segment_measurements


def smooth_changes(changes, window_frames=1):
    """Centered moving average with replicated endpoints; odd window widths only."""
    if not isinstance(window_frames, int) or window_frames < 1 or window_frames % 2 != 1:
        raise ValueError("Smoothing window must be a positive odd integer")
    if window_frames == 1 or not changes.numel():
        return changes
    padded = F.pad(changes[None, None], (window_frames // 2, window_frames // 2), mode="replicate")
    return F.avg_pool1d(padded, window_frames, stride=1)[0, 0]


def measure_segments(
    embeddings, changes, *, threshold, min_distance_frames, frame_hz, duration_seconds, internal_l2=None
):
    """Use hop-grid internal boundaries and the real audio endpoint.

    The frame grid starts at zero; these are segmentation coordinates, not
    claims about receptive-field centers. Only the last segment absorbs the
    frontend's edge truncation/padding. Edge segments are explicitly flagged.
    """
    if not math.isfinite(frame_hz) or frame_hz <= 0 or not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("Frame rate and audio duration must be finite and positive")
    if changes.shape != (len(embeddings) - 1,):
        raise ValueError("Change signal must have one value per adjacent frame pair")
    boundaries = detect_boundaries(changes, threshold=threshold, min_distance_frames=min_distance_frames)
    if boundaries.numel() and float(boundaries[-1]) / frame_hz >= duration_seconds:
        raise ValueError("Encoder frame grid extends past the audio endpoint")
    rows = segment_measurements(boundaries, num_frames=len(embeddings), frame_hop_seconds=1 / frame_hz)
    raw_changes = change_signal(embeddings, "l2") if internal_l2 is None else internal_l2
    if raw_changes.shape != changes.shape:
        raise ValueError("Internal L2 changes must match the detection signal shape")
    for row in rows:
        start, end = row["start_frame"], row["end_frame"]
        if end == len(embeddings):
            row["end_seconds"] = duration_seconds
        row["duration_ms"] = (row["end_seconds"] - row["start_seconds"]) * 1000
        internal = raw_changes[start : end - 1]
        row.update(
            left_strength=float(changes[start - 1]) if start else None,
            right_strength=float(changes[end - 1]) if end < len(embeddings) else None,
            mean_internal_l2=float(internal.mean()) if internal.numel() else 0.0,
            path_length_l2=float(internal.sum()),
            net_displacement_l2=float(torch.linalg.vector_norm(embeddings[end - 1] - embeddings[start])),
        )
    for index, row in enumerate(rows):
        row["prev_duration_ms"] = rows[index - 1]["duration_ms"] if index else None
        row["next_duration_ms"] = rows[index + 1]["duration_ms"] if index + 1 < len(rows) else None
    return rows
