from typing import TypeVar, TypeAlias

from .nac import NACTarget
from .knnvc import KNNVCTarget

ASRBNTarget: TypeAlias = str

T_Target = TypeVar("T_Target")


__all__ = ["T_Target", "ASRBNTarget", "KNNVCTarget", "NACTarget"]
