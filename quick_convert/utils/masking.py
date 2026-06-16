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

    idx = torch.arange(max_length, device=frame_lengths.device)  # (T,)
    return idx.unsqueeze(0) < frame_lengths.unsqueeze(1)  # (B, T)


def masked_loss(loss_fn, preds, targets, mask, reduction: Literal["frame", "batch_by_sample", "sample"] = "frame"):
    """
    NOTE: `preds` and `targets` should be [B, T, C]
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


def trim_to_min(tensor_a, tensor_b, lengths_a, lengths_b, time_dim=-1, strict=True, max_diff=1):
    """
    Take 2 feature tensors and their lengths, and trim them to the smallest
    values. This is necessary when feature extractors with the same time resolution
    operate on the same audio. Although the output lengths should match up, some extractors have different padding
    and rounding rules
    `strict`: Raise an error if the difference is greater than `max_diff` for any sample.
    `max_diff`: Max tolerated per-sample length difference under strict mode. Defaults to 1
        (same-family extractors). Cross-family pairs (e.g. DAC content vs emo2vec) frame
        audio differently and legitimately differ by a couple of frames, so pass a larger value.
    """

    if strict:
        diffs = torch.abs(lengths_a - lengths_b)
        if (diffs > max_diff).any():
            raise RuntimeError(
                f"`strict` mode doesn't allow length differences greater than {max_diff} for any sample. Found length differences={diffs.tolist()}"
            )

    time_dim_a = time_dim % tensor_a.ndim
    time_dim_b = time_dim % tensor_b.ndim

    T_min = min(tensor_a.shape[time_dim_a], tensor_b.shape[time_dim_b])

    output_a = tensor_a.narrow(dim=time_dim_a, start=0, length=T_min)
    output_b = tensor_b.narrow(dim=time_dim_b, start=0, length=T_min)

    lengths = torch.minimum(lengths_a, lengths_b).clamp(max=T_min)

    return output_a, output_b, lengths
