from abc import ABC, abstractmethod

import torch
from torch import nn

from quick_convert.utils.device import DeviceLike, configure_device


class BaseDecoder(nn.Module, ABC):
    def __init__(
        self,
        device: DeviceLike = None,
    ):
        super().__init__()
        self.device = configure_device(device)

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
