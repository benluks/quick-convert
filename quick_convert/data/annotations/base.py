from pathlib import Path
from typing import Any


class BaseAnnotationProvider:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, sample: Any) -> Any:
        raise NotImplementedError
