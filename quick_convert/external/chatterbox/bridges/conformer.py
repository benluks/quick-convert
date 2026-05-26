from __future__ import annotations

import copy
from typing import Optional

import torch
from torch import nn

from ..s3gen.matcha.encoder_layer import ConformerEncoderLayer


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        size: int,
        num_layers: int,
        self_attn: nn.Module,
        feed_forward: nn.Module,
        feed_forward_macaron: Optional[nn.Module],
        conv_module: Optional[nn.Module],
        pos_enc: nn.Module,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, size),
            nn.Dropout(dropout_rate),
        )

        self.pos_enc = pos_enc

        self.layers = nn.ModuleList(
            [
                ConformerEncoderLayer(
                    size=size,
                    self_attn=copy.deepcopy(self_attn),
                    feed_forward=copy.deepcopy(feed_forward),
                    feed_forward_macaron=copy.deepcopy(feed_forward_macaron)
                    if feed_forward_macaron is not None
                    else None,
                    conv_module=copy.deepcopy(conv_module) if conv_module is not None else None,
                    dropout_rate=dropout_rate,
                    normalize_before=normalize_before,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(size, eps=1e-12)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_dim)
            lengths: optional (B,) valid sequence lengths

        Returns:
            x: (B, T, size)
            mask_pad: (B, 1, T)
        """

        x = self.input_proj(x)

        mask_pad = self.make_pad_mask(x, lengths)  # (B, 1, T)
        attn_mask = mask_pad.transpose(1, 2) & mask_pad  # (B, T, T)

        x, pos_emb = self.pos_enc(x)

        for layer in self.layers:
            x, attn_mask, _, _ = layer(
                x=x,
                mask=attn_mask,
                pos_emb=pos_emb,
                mask_pad=mask_pad,
            )

        x = self.norm(x)

        return x, mask_pad

    @staticmethod
    def make_pad_mask(
        x: torch.Tensor,
        lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns mask where True means valid / non-padding.

        Shape: (B, 1, T)
        """
        batch_size, max_len, _ = x.shape

        if lengths is None:
            return torch.ones(
                batch_size,
                1,
                max_len,
                dtype=torch.bool,
                device=x.device,
            )

        idx = torch.arange(max_len, device=x.device).unsqueeze(0)
        mask = idx < lengths.unsqueeze(1)

        return mask.unsqueeze(1)
