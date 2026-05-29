from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import torch
from torch.nn.utils.rnn import pad_sequence


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

    def __setitem__(self, name: str, ref: ResourceRef) -> None:
        self._items[name] = ref

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


@dataclass(frozen=True)
class TensorResourceBatch:
    values: torch.Tensor
    lengths: torch.Tensor


def _normalize_tensor_resource(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensor resources to [T, D].

    Accepted:
    - [D]       -> [1, D]
    - [1, D]    -> [1, D]
    - [T, D]    -> [T, D]
    - [1,T,D]   -> [T, D]
    """
    if x.dim() == 1:
        return x.unsqueeze(0)

    if x.dim() == 2:
        return x

    if x.dim() == 3 and x.shape[0] == 1:
        return x.squeeze(0)

    raise ValueError(f"Expected tensor resource with shape [D], [1,D], [T,D], or [1,T,D]. Got shape {tuple(x.shape)}.")


def _collate_tensor_resources(
    refs: list[ResourceRef],
    squeeze_single_frame: bool = False,
) -> TensorResourceBatch:
    tensors = []
    for ref in refs:
        if ref.value is None:
            raise ValueError(f"Resource {ref.name} has no loaded value. Make sure it is included in dataset.load.")
        if not isinstance(ref.value, torch.Tensor):
            raise TypeError(
                f"Resource {ref.name} has kind='torch_tensor' but value is "
                f"{type(ref.value).__name__}, not torch.Tensor."
            )

        tensors.append(_normalize_tensor_resource(ref.value))

    lengths = torch.tensor([x.shape[0] for x in tensors], dtype=torch.long)
    padded = pad_sequence(tensors, batch_first=True)

    if squeeze_single_frame and padded.shape[1] == 1:
        padded = padded.squeeze(1)

    return TensorResourceBatch(values=padded, lengths=lengths)


def _collate_resource_refs(
    refs: list[ResourceRef],
    squeeze_single_frame_tensors: bool = False,
) -> Any:
    kinds = {ref.kind for ref in refs}
    if len(kinds) != 1:
        raise ValueError(f"Cannot collate mixed resource kinds: {sorted(kinds)}")

    kind = refs[0].kind

    if kind == "text":
        return [ref.value for ref in refs]

    if kind == "torch_tensor":
        return _collate_tensor_resources(
            refs,
            squeeze_single_frame=squeeze_single_frame_tensors,
        )

    raise NotImplementedError(f"Collation for resource kind {kind!r} is not implemented.")


def collate_resources(
    batch,
    squeeze_single_frame_tensors: bool = False,
) -> dict[str, Any]:
    resource_names = {name for item in batch for name in (item.resources.keys() if item.resources is not None else [])}

    collated = {}

    for name in resource_names:
        refs = []
        for item in batch:
            if item.resources is None or name not in item.resources.keys():
                raise ValueError(f"Sample {item.utt_id!r} is missing resource {name!r}")
            refs.append(item.resources[name])

        collated[name] = _collate_resource_refs(
            refs,
            squeeze_single_frame_tensors=squeeze_single_frame_tensors,
        )

    return collated
