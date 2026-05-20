from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class LinearEncoder(nn.Module):
    """
    Minimal placeholder encoder.

    Input:
        x:      [B, T, input_size]
        x_lens: [B]

    Output:
        y:      [B, T, output_size]
        mask:   [B, 1, T]

    Intended as a drop-in replacement for the UpsampleConformerEncoder
    while prototyping.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self._output_size = output_size

        self.proj = nn.Linear(
            input_size,
            output_size,
            bias=bias,
        )

    def output_size(self) -> int:
        return self._output_size

    def make_mask(
        self,
        lengths: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        """
        Returns:
            mask: [B, 1, T]
        """
        idx = torch.arange(
            max_len,
            device=lengths.device,
        )

        mask = idx.unsqueeze(0) < lengths.unsqueeze(1)

        return mask.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:
                [B, T, input_size]

            x_lens:
                [B]

        Returns:
            y:
                [B, T, output_size]

            mask:
                [B, 1, T]
        """

        y = self.proj(x)

        mask = self.make_mask(
            x_lens,
            max_len=y.size(1),
        )

        y = y * mask.transpose(1, 2).to(y.dtype)

        return y, mask
