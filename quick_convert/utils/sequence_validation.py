"""Shared structural checks for padded batch-first sequences.

The padded-length check is structural; the feature-sequence check adds the
nonempty model-input contract. Prefix, waveform, and clock semantics stay separate.
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


def validate_feature_sequence(
    values: torch.Tensor,
    lengths: torch.Tensor,
    *,
    allow_layers: bool = False,
    feature_dim: int | None = None,
    name: str = "Content",
) -> None:
    """Validate nonempty batch/time/feature storage at a model boundary.

    An optional layer axis may precede the feature axis before layer fusion.
    Lengths must be integer tensors; empty prefixes use a different contract.
    """
    if not isinstance(values, torch.Tensor) or not isinstance(lengths, torch.Tensor):
        raise TypeError(f"{name} values and lengths must be tensors.")
    if values.ndim not in ((3, 4) if allow_layers else (3,)) or any(size == 0 for size in values.shape):
        layout = "[batch, time, (layers), feature]" if allow_layers else "[batch, time, feature]"
        raise ValueError(f"{name} features must have nonempty shape {layout}, got {tuple(values.shape)}.")
    if lengths.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise ValueError(f"{name} lengths must use an integer dtype, got {lengths.dtype}.")
    validate_padded_lengths(values, lengths, name=name)
    if torch.any(lengths == 0):
        raise ValueError(f"{name} lengths must be positive.")
    if feature_dim is not None and values.shape[-1] != feature_dim:
        raise ValueError(f"{name} feature dimension must be {feature_dim}, got {values.shape[-1]}.")
