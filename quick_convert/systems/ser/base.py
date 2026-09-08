from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class SERSystem(torch.nn.Module, ABC):
    # def __init__(self, name: str, device: str):
    #     super().__init__()
    #     self.device = device
    #     self.name = name

    @abstractmethod
    def get_labels(self, batch: Any) -> list:
        pass
