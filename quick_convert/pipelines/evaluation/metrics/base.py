from abc import ABC, abstractmethod
from typing import Any, Iterable

from ....data.types import AudioBatch


class Metric(ABC):
    @abstractmethod
    def compute(self, references: list[str], hypotheses: list[str]) -> dict[str, int | float]:
        """
        Run an evaluation on a batch of hypotheses and their respective references, return a dict with metric values
        (indexed by their names)
        """
        ...

    def get_references(self, batch: AudioBatch) -> Iterable[Any]:
        """
        Get task-specific references
        """
        ...
