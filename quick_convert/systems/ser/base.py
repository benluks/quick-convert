from abc import ABC, abstractmethod
from typing import Any

import torch


class SERSystem(torch.nn.Module, ABC):
    """Base interface for speech-emotion recognition systems."""

    # def __init__(self, name: str, device: str):
    #     super().__init__()
    #     self.device = device
    #     self.name = name

    @abstractmethod
    def get_labels(self, batch: Any) -> list:
        """Return one emotion prediction or representation per input sample."""
        raise NotImplementedError
