from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class HeadTarget:
    """
    Supervision passed to a prediction head.

    `lengths` represents target lengths when the target is sequential.
    For class labels or fixed-size embeddings, it can remain None.
    """

    values: Any
    lengths: torch.Tensor | None = None


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
