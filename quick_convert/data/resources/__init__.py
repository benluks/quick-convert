"""
resources allow you to pass arbitrary metadata or sidecar files associated with each sample in the dataset. This can include metadata (anotations), features, or any path.
"""

from .annotations import CSVTranscriptProvider
from .base import Annotation, BaseResourceProvider, ResourceCollection, ResourceRef, collate_resources
from .loader import load_resource
from .providers import PathResourceProvider, TemplateResourceProvider


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
