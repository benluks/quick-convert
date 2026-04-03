from typing import TypeVar

from .nac import NACTarget
from .knnvc import KNNVCTarget

T_Target = TypeVar("T_Target")


__all__ = ["T_Target", "NACTarget", "KNNVCTarget"]
