"""Composable resources associated with dataset samples.

A resource is any named value associated with an utterance, including
transcripts, labels, embeddings, token sequences, SSL features, or acoustic
measurements.

Providers resolve resources for samples, ``ResourceRef`` describes either an
in-memory value or a path-backed value, loaders materialize path-backed
resources, and collation converts sample-level resources into batch-level
representations.
"""

from .base import (
    ResourceCollection,
    ResourceRef,
    collate_resources,
)
from .factory import load_resource_provider
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
    "load_resource_provider",
]
