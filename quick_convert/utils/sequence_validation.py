"""Shared structural checks for padded batch-first sequences.

Semantic constraints (dtype, nonempty inputs, devices, clocks) belong to callers.
"""

import torch


def validate_padded_lengths(
    values: torch.Tensor,
    lengths: torch.Tensor,
    *,
    time_dim: int = 1,
    name: str = "Sequence",
) -> None:
    """Require one nonnegative valid length per item, bounded by padded time."""
    if lengths.shape != (values.shape[0],):
        raise ValueError(f"{name} lengths must have shape ({values.shape[0]},), got {tuple(lengths.shape)}.")
    if torch.any(lengths < 0):
        raise ValueError(f"{name} lengths must be non-negative.")
    if torch.any(lengths > values.shape[time_dim]):
        raise ValueError(f"{name} length exceeds the padded time dimension.")
