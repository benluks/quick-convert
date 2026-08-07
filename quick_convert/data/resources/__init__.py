"""
resources allow you to pass arbitrary metadata or sidecar files associated with each sample in the dataset. This can include metadata (anotations), features, or any path.
"""

from .base import (
    ResourceCollection,
    ResourceRef,
    collate_resources,
)
from .loading import load_resource
from .providers import (
    BaseResourceProvider,
    CSVAnnotationProvider,
    OnlineResourceProvider,
    PathResourceProvider,
    TemplateResourceProvider,
)


__all__ = [
    "BaseResourceProvider",
    "CSVAnnotationProvider",
    "OnlineResourceProvider",
    "PathResourceProvider",
    "ResourceCollection",
    "ResourceRef",
    "TemplateResourceProvider",
    "collate_resources",
    "load_resource",
]
