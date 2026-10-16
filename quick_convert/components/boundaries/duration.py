"""Continuous SSL composition and categorical frame-duration probe components."""

import torch
import torch.nn.functional as F
from torch import nn


def composition_moments(trajectory, points=16):
    """Mean and population SD on a fixed relative-time grid, without timestamps.

    This removes explicit frame count from the descriptor, not every possible
    timing cue in SSL values or interpolation. No classes are assigned to content.
    """
    if (
        trajectory.ndim != 2
        or len(trajectory) < 2
        or not trajectory.is_floating_point()
        or not torch.isfinite(trajectory).all()
    ):
        raise ValueError("Expected a finite floating-point [T,D] segment with T>=2")
    if not isinstance(points, int) or points < 2:
        raise ValueError("points must be an integer >=2")
    normalized = F.interpolate(trajectory.T[None].float(), size=points, mode="linear", align_corners=True)[0]
    return torch.stack((normalized.mean(-1), normalized.std(-1, correction=0)))


def fit_frame_bins(training_frames, n_bins=12):
    """Fit inclusive integer upper edges using training quantiles only.

    Duplicate edges from tied integer durations collapse; the last bin includes
    the training maximum and any longer held-out durations.
    """
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins must be >=2")
    if (
        training_frames.ndim != 1
        or not len(training_frames)
        or training_frames.is_floating_point()
        or torch.any(training_frames < 1)
    ):
        raise ValueError("Expected nonempty positive integer training frame counts")
    quantiles = torch.arange(1, n_bins, dtype=torch.float64) / n_bins
    edges = torch.quantile(training_frames.double(), quantiles).ceil().long().unique(sorted=True)
    return edges[edges < training_frames.max()]


def duration_labels(frames, edges):
    """Map frame counts to ordered bins; equality belongs to the lower bin."""
    return torch.bucketize(frames.contiguous(), edges.to(frames.device), right=False)


def descriptor_slice(features, mode):
    """Select mean, variation, or their concatenation from [N,2,P] sketches."""
    if mode == "mean":
        return features[:, 0]
    if mode == "variation":
        return features[:, 1]
    if mode == "both":
        return features.flatten(1)
    raise ValueError(f"Unknown descriptor mode: {mode}")


class CategoricalDurationProbe(nn.Module):
    """Small probability head over duration bins; SSL content stays continuous."""

    def __init__(self, input_dim, n_bins, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_bins)
        )

    def forward(self, features):
        return self.network(features)
