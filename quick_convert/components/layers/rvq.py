from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class RVQLosses:
    """Losses produced by a residual vector quantizer.

    ``loss`` is the scalar objective consumed by training modules. ``raw``
    contains the unweighted component losses, while ``weighted`` contains the
    corresponding contributions to ``loss``.
    """

    loss: torch.Tensor
    raw: dict[str, torch.Tensor] = field(default_factory=dict)
    weighted: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class RVQOutput:
    """Common output contract for residual vector quantizers.

    Shapes:
        z_q: ``[B, D, T]`` summed quantized representation.
        layer_z_qs: per-codebook additive contributions, each ``[B, D, T]``.
        codes: codebook indices ``[B, Q, T]``.
        latents: projected residual inputs ``[B, Q * codebook_dim, T]``.
    """

    z_q: torch.Tensor
    layer_z_qs: list[torch.Tensor]
    codes: torch.Tensor
    latents: torch.Tensor
    loss: RVQLosses


class BaseResidualVectorQuantizer(nn.Module, ABC):
    """Interface shared by residual vector quantizer implementations."""

    input_dim: int
    n_codebooks: int
    codebook_size: int
    codebook_dim: int
    quantizers: nn.ModuleList

    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        padding_mask: torch.Tensor,
        n_quantizers: int | None = None,
    ) -> RVQOutput:
        """Quantize ``z`` with shape ``[B, D, T]`` over valid timesteps."""
