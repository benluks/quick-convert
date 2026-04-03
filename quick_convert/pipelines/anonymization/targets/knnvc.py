from dataclasses import dataclass
from os import PathLike
from typing import Optional


@dataclass(frozen=True)
class KNNVCDirectoryTarget:
    target: PathLike
    pattern: Optional[str] = None


KNNVCTarget = KNNVCDirectoryTarget
