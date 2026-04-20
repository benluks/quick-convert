from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseSpeakerGenerator(nn.Module, ABC):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    @abstractmethod
    def compute_loss(
        self,
        embeddings: torch.Tensor,
        **conditions,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
        **conditions,
    ) -> torch.Tensor:
        raise NotImplementedError