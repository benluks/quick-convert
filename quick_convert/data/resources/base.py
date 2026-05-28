from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional


class BaseResourceProvider:
    """
    An abstracton class for resource providers, which are responsible for providing access to various types of
    resources (e.g. annotation files, precompute feature files, etc.) associated with samples in a dataset.
    """

    def __init__(self, name: str):
        self.name = name

    def __call__(self, sample):
        raise NotImplementedError


ResourceKind = Literal[
    # serialized tensors/features
    "torch_tensor",
    "numpy_array",
    # raw media
    "audio",
    "image",
    "video",
    # structured/textual data
    "text",
    "json",
    "csv",
    # model-specific semantic categories
    "ssl_features",
    "speaker_embedding",
    "prosody",
    "token_ids",
]


@dataclass(frozen=True)
class ResourceRef:
    path: Path
    kind: ResourceKind
    name: str
    value: Optional[Any] = None
    # metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Annotation(ResourceRef):
    """
    Since annotations are automatically loaded into memory when accessed,
    we can also include the annotation value directly in the object for convenience.
    This abstraction also lets us distinguish which resources need to be loaded at runtime
    """

    value: Any
    path: Path


@dataclass
class ResourceCollection:
    _items: dict[str, ResourceRef] = field(default_factory=dict)

    def __getitem__(self, name: str) -> ResourceRef:
        return self._items[name]

    def __getattr__(self, name: str) -> ResourceRef:
        try:
            return self._items[name]
        except KeyError:
            raise AttributeError(name) from None

    def __iter__(self):
        return iter(self._items.values())

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()

    def items(self):
        return self._items.items()

    def as_dict(self) -> dict[str, ResourceRef]:
        return dict(self._items)

    @classmethod
    def from_refs(cls, refs: Iterable[ResourceRef]) -> "ResourceCollection":
        items = {}
        for ref in refs:
            if ref.name in items:
                raise ValueError(f"Duplicate resource name: {ref.name}")
            items[ref.name] = ref
        return cls(items)
