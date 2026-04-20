# components/feature_extractors/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseFeatureExtractor(ABC):
    @property
    @abstractmethod
    def feature_name(self) -> str:
        """Name used for saving and directory structure."""
        ...

    @abstractmethod
    def extract_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: output of DataLoader collate_fn

        Returns:
            dict of batched features, e.g.:
                {"embedding": Tensor[B, D]}
        """
        ...
