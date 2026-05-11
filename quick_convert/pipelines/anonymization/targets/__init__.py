try:
    from typing import TypeVar, TypeAlias
except ImportError:
    # version resolution for python 3.9 vs 3.10+
    from typing import TypeVar
    from typing_extensions import TypeAlias

from .nac import NACTarget
from .knnvc import KNNVCTarget

ASRBNTarget: TypeAlias = str

T_Target = TypeVar("T_Target")


__all__ = ["T_Target", "ASRBNTarget", "KNNVCTarget", "NACTarget"]
