"""Shared content adaptation for inference systems; modules remain caller-owned."""

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from quick_convert.components.layers.rvq import RVQOutput
from quick_convert.utils.masking import make_padding_mask
from quick_convert.utils.sequence_validation import validate_feature_sequence


def module_output_lengths(module: nn.Module, lengths: torch.Tensor, role: str) -> torch.Tensor:
    if isinstance(module, nn.Identity):
        return lengths
    output_lengths = getattr(module, "output_lengths", None)
    if not callable(output_lengths):
        raise TypeError(f"{role} modules must expose output_lengths(input_lengths).")
    return output_lengths(lengths)


def transform_content(
    values: torch.Tensor,
    lengths: torch.Tensor,
    transform: nn.Module,
    *,
    metadata: Mapping[str, Any] | None = None,
    trim: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_feature_sequence(values, lengths, allow_layers=True)
    if not values.is_floating_point():
        raise TypeError("Continuous content systems require floating-point features.")
    if metadata is not None:
        content_frame_rate(metadata)
        if metadata.get("feature_dim", values.shape[-1]) != values.shape[-1]:
            raise ValueError("Content metadata feature_dim differs from feature storage.")
        if metadata.get("temporal_granularity", "frame") != "frame":
            raise ValueError("Content systems require frame-level representations.")
        if metadata.get("representation_type", "continuous") not in {"continuous", "quantized"}:
            raise ValueError("Content systems require continuous or quantized vectors, not token IDs.")
    features = transform(values)
    output_lengths = module_output_lengths(transform, lengths, "Content transform")
    validate_feature_sequence(features, output_lengths, name="Transformed content")
    if not features.is_floating_point():
        raise TypeError("Content transforms must return floating-point features.")
    if features.shape[0] != values.shape[0]:
        raise ValueError("Content transforms must preserve batch size.")
    if trim:
        features = features[:, : int(output_lengths.max())]
    return features, output_lengths


def quantize_content(
    features: torch.Tensor, lengths: torch.Tensor, quantizer: nn.Module
) -> tuple[RVQOutput, torch.Tensor]:
    validate_feature_sequence(features, lengths, feature_dim=getattr(quantizer, "input_dim", None))
    mask = make_padding_mask(lengths, max_length=features.shape[1])
    output = quantizer(features.transpose(1, 2), mask)
    if not isinstance(output, RVQOutput):
        raise TypeError("Content quantizers must return RVQOutput.")
    output_lengths = module_output_lengths(quantizer, lengths, "Content quantizer")
    for name, values, dim in (
        ("Quantized content", output.z_q, features.shape[-1]),
        ("Quantizer latents", output.latents, None),
    ):
        if not isinstance(values, torch.Tensor) or values.ndim != 3:
            raise ValueError(f"{name} must have shape [batch, feature, time].")
        validate_feature_sequence(values.transpose(1, 2), output_lengths, feature_dim=dim, name=name)
        if values.shape[0] != features.shape[0]:
            raise ValueError(f"{name} must preserve batch size.")
    return output, output_lengths


def content_frame_rate(metadata: Mapping[str, Any] | None, *modules: nn.Module | None) -> float | None:
    """Retain a known clock only through explicitly time-preserving stages.

    Valid output lengths alone do not establish uniform temporal spacing.
    """
    rate = metadata.get("frame_hz") if metadata is not None else None
    if rate is not None and (not math.isfinite(rate) or rate <= 0):
        raise ValueError("Content metadata frame_hz must be finite and positive.")
    for module in modules:
        if (
            module is not None
            and not isinstance(module, nn.Identity)
            and not getattr(module, "PRESERVES_FRAME_RATE", False)
        ):
            return None
    return rate
