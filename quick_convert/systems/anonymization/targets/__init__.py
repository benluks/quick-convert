from typing import TypeAlias, TypeVar

from .knnvc import KNNVCTarget


ASRBNTarget: TypeAlias = str
T_Target = TypeVar("T_Target")


__all__ = ["ASRBNTarget", "KNNVCTarget", "T_Target"]
