"""
resources allow you to pass arbitrary metadata or sidecar files associated with each sample in the dataset. This can include metadata (anotations), features, or any path.
"""

from .base import BaseResourceProvider
from .paths import PathResourceProvider
from .annotations import CSVTranscriptProvider

__all__ = [
    "BaseResourceProvider",
    "PathResourceProvider",
    "CSVTranscriptProvider",
]
