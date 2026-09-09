from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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
    "torch_tensor",
    "text",
    "token_ids",
]
RESOURCE_KINDS = frozenset(ResourceKind.__args__)


@dataclass(frozen=True)
class ResourceRef:
    name: str
    kind: ResourceKind
    path: Path | None = None
    value: Any | None = None

    # Only set if using cudnn benchmark
    max_length: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in RESOURCE_KINDS:
            raise ValueError(f"Unsupported resource kind {self.kind!r}. Supported kinds: {sorted(RESOURCE_KINDS)}")
        if self.path is None and self.value is None:
            raise ValueError(f"Resource {self.name!r} must have a path or a value.")
        if self.max_length is not None and self.kind not in {"torch_tensor", "token_ids"}:
            raise ValueError("max_length is only supported for tensor and token resources.")


@dataclass(frozen=True)
class ResourceCollection:
    _items: dict[str, ResourceRef] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_items", dict(self._items))

    def __getitem__(self, name: str) -> ResourceRef:
        return self._items[name]

    def __getattr__(self, name: str) -> ResourceRef:
        if name.startswith("__") or name == "_items":
            raise AttributeError(name)

        items = self.__dict__.get("_items")
        if items is None:
            raise AttributeError(name)

        try:
            return items[name]
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

    def merge(
        self,
        other: "ResourceCollection",
        overwrite: bool = False,
    ) -> "ResourceCollection":
        items = self.as_dict()

        for name, ref in other.items():
            if name in items and not overwrite:
                raise ValueError(f"Duplicate resource name: {name}")
            items[name] = ref

        return ResourceCollection(items)


@dataclass
class TensorResourceBatch:
    values: torch.Tensor
    lengths: torch.Tensor

    def __len__(self) -> int:
        return self.values.shape[0]

    def __getitem__(self, idx: int):
        if self.lengths is None:
            return self.values[idx]

        length = self.lengths[idx]

        # optional: trim padded time dimension
        return self.values[idx, :length]


def _normalize_tensor_resource(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensor resources so the first dim is always time.

    Accepted:
    - [D]         -> [1, D]
    - [T, D]      -> [T, D]
    - [1, T, D]   -> [T, D]
    - [T, L, D]   -> [T, L, D]
    - [1,T,L,D]   -> [T, L, D]
    """
    if x.dim() == 1:
        return x.unsqueeze(0)

    if x.dim() == 2:
        return x

    if x.dim() == 3:
        if x.shape[0] == 1:
            return x.squeeze(0)  # [1, T, D] -> [T, D]
        return x  # [T, L, D]

    if x.dim() == 4 and x.shape[0] == 1:
        return x.squeeze(0)  # [1, T, L, D] -> [T, L, D]

    raise ValueError(
        f"Expected tensor resource with shape [D], [T,D], [1,T,D], [T,L,D], or [1,T,L,D]. Got shape {tuple(x.shape)}."
    )


def _collate_tensor_resources(
    refs: list[ResourceRef], squeeze_single_frame: bool = False, max_length: int | None = None
) -> TensorResourceBatch:
    """
    max_length: An optional arbitrary max length to pad or trim batch. Useful in the case of cudnn, which needs
    all batches to have the same input shape.
    """

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

    trailing_shape = tensors[0].shape[1:]

    for x in tensors[1:]:
        if x.shape[1:] != trailing_shape:
            raise ValueError(
                f"Cannot collate tensor resources with mismatched trailing shapes: {trailing_shape} vs {x.shape[1:]}"
            )

    lengths = torch.tensor([x.shape[0] for x in tensors], dtype=torch.long)
    if max_length is not None:
        if lengths.max() > max_length:
            raise ValueError(f"Resource max_length={max_length} is shorter than a sequence in the batch.")
        tensors[0] = torch.nn.functional.pad(
            tensors[0],
            (0, 0) * (tensors[0].dim() - 1) + (0, max_length - int(lengths[0])),
        )

    padded = pad_sequence(tensors, batch_first=True)

    if squeeze_single_frame and padded.shape[1] == 1:
        padded = padded.squeeze(1)

    return TensorResourceBatch(values=padded, lengths=lengths)


def _collate_resource_refs(refs: list[ResourceRef], squeeze_single_frame_tensors: bool = False) -> Any:
    kinds = {ref.kind for ref in refs}
    if len(kinds) != 1:
        raise ValueError(f"Cannot collate mixed resource kinds: {sorted(kinds)}")

    kind = refs[0].kind
    max_lengths = {ref.max_length for ref in refs}
    if len(max_lengths) != 1:
        raise ValueError(
            f"Resource {refs[0].name!r} has inconsistent max_length values: {sorted(max_lengths, key=str)}"
        )
    max_length = refs[0].max_length

    if kind == "text":
        return [ref.value for ref in refs]

    if kind == "torch_tensor":
        return _collate_tensor_resources(
            refs,
            squeeze_single_frame=squeeze_single_frame_tensors,
            max_length=max_length,
        )
    if kind == "token_ids":
        return collate_token_sequences([ref.value for ref in refs], padding_value=0, max_length=max_length)

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

        collated[name] = _collate_resource_refs(refs, squeeze_single_frame_tensors=squeeze_single_frame_tensors)

    return collated


def collate_token_sequences(
    sequences: list[list[int] | torch.Tensor],
    padding_value: int = 0,
    max_length: int | None = None,
) -> TensorResourceBatch:
    tensors = [torch.as_tensor(seq, dtype=torch.long) for seq in sequences]

    lengths = torch.tensor(
        [len(x) for x in tensors],
        dtype=torch.long,
    )

    if max_length is not None:
        if lengths.max() > max_length:
            raise ValueError(f"Resource max_length={max_length} is shorter than a token sequence in the batch.")
        tensors[0] = torch.nn.functional.pad(
            tensors[0],
            (0, max_length - int(lengths[0])),
            value=padding_value,
        )

    padded = pad_sequence(
        tensors,
        batch_first=True,
        padding_value=padding_value,
    )

    return TensorResourceBatch(values=padded, lengths=lengths)
