from __future__ import annotations

from abc import ABC, abstractmethod


class BaseResourceExtractor(ABC):
    @property
    def key(self) -> str:
        return self.feature_name

    @property
    @abstractmethod
    def feature_name(self) -> str:
        """Name of the resource produced by this adapter."""
        ...

    def extract_sample(self, sample):
        raise NotImplementedError

    def extract_batch(self, batch):
        """Adapt sample-only implementations; batch implementations may override."""
        return [self.extract_sample(sample) for sample in batch]

    def provide_sample(self, sample):
        return self.extract_sample(sample)

    def provide_batch(self, batch):
        return self.extract_batch(batch)
