from dataclasses import dataclass
from os import PathLike


@dataclass(frozen=True)
class KNNVCDirectoryTarget:
    target: PathLike
    pattern: str | None = None


KNNVCTarget = KNNVCDirectoryTarget
