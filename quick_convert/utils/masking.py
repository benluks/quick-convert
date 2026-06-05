from typing import Literal, Optional

import torch


def mask_pad(x, padding_mask):
    """
    Padding mask, False means pad
    """
    return x.masked_fill(~padding_mask.unsqueeze(-1), 0.0)


def make_padding_mask(frame_lengths: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """Returns (B, T) bool mask — False marks padding positions."""

    max_length = max_length or frame_lengths.max()

    idx = torch.arange(frame_lengths.max(), device=frame_lengths.device)  # (T,)
    return idx.unsqueeze(0) < frame_lengths.unsqueeze(1)  # (B, T)


def masked_loss(loss_fn, preds, targets, mask, reduction: Literal["frame", "batch_by_sample", "sample"] = "frame"):
    """
    reduction:
      "frame"  = scalar, average over all valid elements in batch
      "batch_by_sample" = scalar, average per sample, then across batch
      "sample"   = (B,), masked mean per sample
    """
    loss = loss_fn(preds, targets, reduction="none")

    if mask.ndim == loss.ndim - 1:
        mask = mask.unsqueeze(-1)

    mask = mask.to(loss.device, dtype=loss.dtype)

    loss = loss * mask

    reduce_dims = tuple(range(1, loss.ndim))

    if reduction == "frame":
        return loss.sum() / mask.sum().clamp_min(1.0)

    if reduction == "batch_by_sample":
        per_sample = loss.sum(dim=reduce_dims) / mask.sum(dim=reduce_dims).clamp_min(1.0)
        return per_sample.mean()

    if reduction == "sample":
        return loss.sum(dim=reduce_dims) / mask.sum(dim=reduce_dims).clamp_min(1.0)

    raise ValueError(f"Unknown reduction: {reduction}")
