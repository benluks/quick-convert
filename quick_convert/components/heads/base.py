from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from quick_convert.utils.masking import make_padding_mask


@dataclass
class HeadTarget:
    """
    Supervision passed to a prediction head.

    `lengths` represents target lengths when the target is sequential.
    For class labels or fixed-size embeddings, it can remain None.
    Sequential targets may carry a boolean [B,T] observation_mask and aligned
    [B,T] timestamps in seconds. valid_mask combines observations and padding;
    it does not perform temporal alignment.
    """

    values: Any
    lengths: torch.Tensor | None = None
    observation_mask: torch.Tensor | None = None
    timestamps: torch.Tensor | None = None

    def valid_mask(self, prediction_mask: torch.Tensor) -> torch.Tensor:
        """Combine prediction padding, target padding, and observed-frame validity.

        True means usable. This operation assumes an already aligned time grid;
        equal shapes alone do not prove temporal alignment.
        """
        if prediction_mask.ndim == 3 and prediction_mask.shape[-1] == 1:
            prediction_mask = prediction_mask.squeeze(-1)
        if prediction_mask.ndim != 2 or prediction_mask.dtype != torch.bool:
            raise ValueError("Prediction validity must be a boolean [B, T] mask.")
        if not isinstance(self.values, torch.Tensor) or self.values.shape[:2] != prediction_mask.shape:
            raise ValueError("Sequential target and prediction batch/time dimensions must match.")
        mask = prediction_mask.clone()
        if self.lengths is not None:
            if self.lengths.shape != (mask.shape[0],) or self.lengths.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise ValueError("Target lengths must contain one integer per batch item.")
            if torch.any(self.lengths < 0) or torch.any(self.lengths > mask.shape[1]):
                raise ValueError("Target lengths must fit the padded time dimension.")
            mask &= make_padding_mask(self.lengths.to(mask.device), mask.shape[1])
        if self.observation_mask is not None:
            if self.observation_mask.shape != mask.shape or self.observation_mask.dtype != torch.bool:
                raise ValueError("Observation validity must be a boolean [B, T] mask.")
            mask &= self.observation_mask.to(mask.device)
        return mask


@dataclass
class HeadOutput:
    """
    Common output returned by every supervised head.

    `features` contains reusable representations produced by a head, such as
    a speaker embedding used to condition a downstream decoder.
    """

    loss: torch.Tensor | None = None
    predictions: Any = None
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)
    features: dict[str, torch.Tensor] = field(default_factory=dict)
    states: dict[str, Any] = field(default_factory=dict)


class SupervisedHead(nn.Module, ABC):
    @abstractmethod
    def compute_loss(
        self,
        features: torch.Tensor,
        *,
        targets: HeadTarget,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> HeadOutput: ...

    def predict(
        self,
        features: torch.Tensor,
        *,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> HeadOutput:
        raise NotImplementedError
