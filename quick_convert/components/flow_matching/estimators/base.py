from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseFlowEstimator(nn.Module, ABC):
    """
    Base interface for flow/velocity estimators used inside a flow-matching module.

    The estimator predicts a vector field in the target space, typically:
        v_t = f(x_t, t, cond)

    where:
        - x_t: current noised / interpolated state
        - t: scalar time for each batch item
        - cond: conditioning vector
    """

    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        time_dim: int,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.time_dim = time_dim
        self.output_dim = output_dim if output_dim is not None else input_dim

    @abstractmethod
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t:
                Tensor of shape (B, input_dim)
            t:
                Tensor of shape (B,) or (B, 1)
            cond:
                Tensor of shape (B, cond_dim)

        Returns:
            Tensor of shape (B, output_dim)
        """
        raise NotImplementedError
