"""Named annotations and sidecar values attached to dataset samples.

Providers resolve a :class:`ResourceRef` from sample metadata. A dataset then
decides whether to keep the reference lazy or load its value during access.
"""

from .annotations import CSVAnnotationProvider
from .base import BaseResourceProvider, ResourceCollection, ResourceRef, collate_resources
from .loader import load_resource
from .providers import PathResourceProvider, TemplateResourceProvider


__all__ = [
    "BaseResourceProvider",
    "PathResourceProvider",
    "TemplateResourceProvider",
    "CSVAnnotationProvider",
    "ResourceRef",
    "ResourceCollection",
    "load_resource",
    "collate_resources",
]
