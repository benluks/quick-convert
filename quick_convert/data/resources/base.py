from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch.nn.utils.rnn import pad_sequence


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
    "token_ids",
]


@dataclass
class ResourceRef:
    """Description of a named resource associated with one sample.

    A resource may either contain a materialized ``value`` or point to a
    serialized representation through ``path``. Path-backed resources can be
    loaded lazily by :func:`load_resource`.

    ``name`` identifies the resource within a sample, while ``kind`` determines
    how it is loaded and collated.

    Examples:
        An immediately available metadata value::

            ResourceRef(
                name="speaker_id",
                kind="text",
                value="1089",
            )

        A lazily loaded tensor::

            ResourceRef(
                name="wavlm",
                kind="torch_tensor",
                path=Path("/features/1089-134686-0000.pt"),
            )

    Args:
        name:
            Name used to access the resource, such as ``"transcript"`` or
            ``"wavlm"``.
        kind:
            Resource representation used to select loading and collation
            behavior.
        path:
            Optional path to a serialized resource.
        value:
            Optional materialized resource value.
        max_length:
            Optional fixed time length used when collating tensor resources.
    """

    name: str
    kind: ResourceKind | None = None
    path: Path | None = None
    value: Any | None = None

    # Only set if using cudnn benchmark
    max_length: int | None = None


@dataclass
class ResourceCollection:
    """Named collection of sample-level :class:`ResourceRef` objects.

    Resources can be accessed using either dictionary or attribute syntax::

        sample.resources["transcript"]
        sample.resources.transcript

    Resource names must be unique within a collection.
    """

    _items: dict[str, ResourceRef] = field(default_factory=dict)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __getitem__(self, name: str) -> ResourceRef:
        return self._items[name]

    def __setitem__(self, name: str, ref: ResourceRef) -> None:
        self._items[name] = ref

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
        """Construct a collection from resource references.

        Raises:
            ValueError:
                If two resources have the same name.
        """
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
        """Return a new collection containing resources from both collections.

        Args:
            other:
                Resources to add.
            overwrite:
                Whether resources in ``other`` may replace resources with the
                same name.

        Raises:
            ValueError:
                If duplicate names are encountered and ``overwrite`` is false.
        """
        items = self.as_dict()

        for name, ref in other.items():
            if name in items and not overwrite:
                raise ValueError(f"Duplicate resource name: {name}")
            items[name] = ref

        return ResourceCollection(items)


@dataclass
class TensorResourceBatch:
    """Padded batch of variable-length tensor resources.

    ``values`` contains the padded tensors and ``lengths`` records the original
    length of each item along the first, time-like dimension.

    Indexing returns an individual item with padding removed::

        features = batch.resources["wavlm"]
        first_utterance = features[0]
    """

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
    """Normalize a tensor so its leading dimension represents time.

    Accepted shapes::

        [D]          -> [1, D]
        [T, D]       -> [T, D]
        [1, T, D]    -> [T, D]
        [T, L, D]    -> [T, L, D]
        [1, T, L, D] -> [T, L, D]

    This allows frame-level features, fixed embeddings, and multi-layer
    features to share the same collation path.
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
    """Pad tensor resources along their leading time dimension.

    All trailing dimensions must match. The original sequence lengths are
    retained in the returned :class:`TensorResourceBatch`.

    ``max_length`` may be used to force a consistent padded time dimension,
    for example when fixed batch shapes are useful for compilation or cuDNN
    benchmarking.
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
            raise NotImplementedError("This function is only implemented to increase the maximum length. ")
        T, *d_rest = tensors[0].shape
        padded_tensor = torch.zeros((max_length, *d_rest), dtype=tensors[0].dtype)
        padded_tensor[:T] = tensors[0]
        tensors[0] = padded_tensor

    padded = pad_sequence(tensors, batch_first=True)

    if squeeze_single_frame and padded.shape[1] == 1:
        padded = padded.squeeze(1)

    return TensorResourceBatch(values=padded, lengths=lengths)


def _collate_resource_refs(refs: list[ResourceRef], squeeze_single_frame_tensors: bool = False) -> Any:
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
            # max_length is carried by the ResourceRef so fixed-shape padding can
            # remain resource-specific without adding dataset-level special cases.
            max_length=refs[0].max_length,
        )
    if kind == "token_ids":
        return collate_token_sequences([ref.value for ref in refs], padding_value=0)

    raise NotImplementedError(f"Collation for resource kind {kind!r} is not implemented.")


def collate_resources(
    batch,
    squeeze_single_frame_tensors: bool = False,
) -> dict[str, Any]:
    """Collate all resources shared by the samples in a batch.

    Each resource name is collated according to its ``kind``. Every sample
    must contain every resource present in the batch; missing resources raise
    an error rather than silently producing an irregular batch.

    Typical outputs include:

    - ``text`` -> ``list[str]``
    - ``torch_tensor`` -> :class:`TensorResourceBatch`
    - ``token_ids`` -> :class:`TensorResourceBatch`
    """
    resource_names = {name for item in batch for name in (item.resources.keys() if item.resources is not None else [])}

    collated = {}

    for name in resource_names:
        refs = []
        for item in batch:
            if item.resources is None or name not in item.resources:
                raise ValueError(f"Sample {item.utt_id!r} is missing resource {name!r}")
            refs.append(item.resources[name])

        collated[name] = _collate_resource_refs(refs, squeeze_single_frame_tensors=squeeze_single_frame_tensors)

    return collated


def collate_token_sequences(sequences: list[list[int]], padding_value: int = 0) -> TensorResourceBatch:
    """Pad integer token sequences and retain their original lengths."""
    tensors = [torch.tensor(seq, dtype=torch.long) for seq in sequences]

    lengths = torch.tensor(
        [len(x) for x in tensors],
        dtype=torch.long,
    )

    padded = pad_sequence(
        tensors,
        batch_first=True,
        padding_value=padding_value,
    )

    return TensorResourceBatch(values=padded, lengths=lengths)
