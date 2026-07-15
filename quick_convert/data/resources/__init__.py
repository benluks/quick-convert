"""
resources allow you to pass arbitrary metadata or sidecar files associated with each sample in the dataset. This can include metadata (anotations), features, or any path.
"""

from .base import BaseResourceProvider, ResourceRef, ResourceCollection, Annotation, collate_resources
from .providers import PathResourceProvider, TemplateResourceProvider
from .annotations import CSVTranscriptProvider
from .loader import load_resource

__all__ = [
    "BaseResourceProvider",
    "PathResourceProvider",
    "TemplateResourceProvider",
    "CSVTranscriptProvider",
    "ResourceRef",
    "ResourceCollection",
    "Annotation",
    "load_resource",
    "collate_resources",
]
