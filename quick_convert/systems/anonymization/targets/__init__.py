from typing import TypeAlias, TypeVar

from .knnvc import KNNVCTarget
from .speech import PreparedTarget, ReferenceSpeech


ASRBNTarget: TypeAlias = str
T_Target = TypeVar("T_Target")


__all__ = ["ASRBNTarget", "KNNVCTarget", "T_Target", "PreparedTarget", "ReferenceSpeech"]
