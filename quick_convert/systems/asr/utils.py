from __future__ import annotations
from typing import Optional
import torch


def greedy_ctc_decode(
    ids: Optional[int["[1] t"]] = None,
    logits: Optional[int["[1] t v"]] = None,
    blank_id: int = 0,
    temperature: int = 1,
) -> torch.Tensor:
    if ids is None:
        ids = logits.argmax(-1)
    ids = torch.unique_consecutive(ids)
    ids = ids[ids != blank_id]
    return ids
