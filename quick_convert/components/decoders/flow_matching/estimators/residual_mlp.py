from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import BaseFlowEstimator


class TimeEmbedding(nn.Module):
    """
    Simple sinusoidal time embedding followed by an MLP projection.

    Input:
        t: (B,) or (B, 1)

    Output:
        (B, time_dim)
    """

    def __init__(
        self,
        time_dim: int,
        hidden_dim: int | None = None,
        max_period: int = 10_000,
    ) -> None:
        super().__init__()
        if time_dim <= 0:
            raise ValueError(f"time_dim must be > 0, got {time_dim}")

        self.time_dim = time_dim
        self.max_period = max_period

        proj_hidden = hidden_dim if hidden_dim is not None else time_dim * 2

        self.proj = nn.Sequential(
            nn.Linear(time_dim, proj_hidden),
            nn.SiLU(),
            nn.Linear(proj_hidden, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        elif t.ndim != 1:
            raise ValueError(f"Expected t with shape (B,) or (B, 1), got {tuple(t.shape)}")

        half_dim = self.time_dim // 2
        device = t.device
        dtype = t.dtype if torch.is_floating_point(t) else torch.float32

        t = t.to(dtype)

        if half_dim == 0:
            emb = t.unsqueeze(-1)
            return self.proj(emb)

        freq_exponent = -math.log(self.max_period) * torch.arange(half_dim, device=device, dtype=dtype) / half_dim
        freqs = torch.exp(freq_exponent)  # (half_dim,)

        args = t.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.time_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return self.proj(emb)


class ResidualMLPBlock(nn.Module):
    """
    Standard pre-norm residual MLP block.

    Shape:
        (B, hidden_dim) -> (B, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if expansion <= 0:
            raise ValueError(f"expansion must be > 0, got {expansion}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        inner_dim = hidden_dim * expansion

        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class ResidualMLPEstimator(BaseFlowEstimator):
    """
    Residual MLP estimator for vector-field prediction in embedding space.

    It predicts:
        v_t = f(x_t, t, cond)

    using:
        - a learned time embedding
        - concatenation of x_t, t_embed, cond
        - input projection
        - residual MLP trunk
        - output projection

    Expected shapes:
        x_t:  (B, input_dim)
        t:    (B,) or (B, 1)
        cond: (B, cond_dim)

    Output:
        (B, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        time_dim: int = 128,
        num_blocks: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            cond_dim=cond_dim,
            time_dim=time_dim,
            output_dim=output_dim,
        )

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        if not 0.0 <= input_dropout < 1.0:
            raise ValueError(f"input_dropout must be in [0, 1), got {input_dropout}")

        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.time_embedding = TimeEmbedding(time_dim=time_dim)

        fused_dim = input_dim + cond_dim + time_dim

        self.input_proj = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(input_dropout),
        )

        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(
                    hidden_dim=hidden_dim,
                    expansion=expansion,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(x_t=x_t, t=t, cond=cond)

        t_embed = self.time_embedding(t)  # (B, time_dim)
        h = torch.cat([x_t, t_embed, cond], dim=-1)
        h = self.input_proj(h)

        for block in self.blocks:
            h = block(h)

        h = self.output_norm(h)
        return self.output_proj(h)

    def _validate_inputs(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> None:
        if x_t.ndim != 2:
            raise ValueError(f"x_t must have shape (B, D), got {tuple(x_t.shape)}")
        if cond.ndim != 2:
            raise ValueError(f"cond must have shape (B, C), got {tuple(cond.shape)}")
        if t.ndim not in {1, 2}:
            raise ValueError(f"t must have shape (B,) or (B, 1), got {tuple(t.shape)}")

        batch_size = x_t.shape[0]

        if cond.shape[0] != batch_size:
            raise ValueError(f"Batch mismatch: x_t has batch {batch_size}, cond has batch {cond.shape[0]}")

        if t.shape[0] != batch_size:
            raise ValueError(f"Batch mismatch: x_t has batch {batch_size}, t has batch {t.shape[0]}")

        if x_t.shape[1] != self.input_dim:
            raise ValueError(f"x_t feature dim mismatch: expected {self.input_dim}, got {x_t.shape[1]}")

        if cond.shape[1] != self.cond_dim:
            raise ValueError(f"cond feature dim mismatch: expected {self.cond_dim}, got {cond.shape[1]}")
