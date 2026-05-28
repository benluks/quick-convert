# components/feature_extractors/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseFeatureExtractor(ABC):
    @property
    def key(self) -> str:
        return self.feature_name

    @property
    def feature_name(self) -> str:
        """Name used for saving and directory structure."""
        raise NotImplementedError

    def extract_sample(self, sample):
        raise NotImplementedError

    def extract_batch(self, batch):
        raise NotImplementedError

    def provide_sample(self, sample):
        return self.extract_sample(sample)

    def provide_batch(self, batch):
        return self.extract_batch(batch)
