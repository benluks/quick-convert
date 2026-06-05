from typing import Optional

import torch


def mask_pad(x, padding_mask):
    return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)


def make_padding_mask(frame_lengths: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """Returns (B, T) bool mask — False marks padding positions."""

    max_length = max_length or frame_lengths.max()

    idx = torch.arange(frame_lengths.max(), device=frame_lengths.device)  # (T,)
    return idx.unsqueeze(0), frame_lengths.unsqueeze(1)  # (B, T)
