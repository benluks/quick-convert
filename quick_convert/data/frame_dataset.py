from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .base_dataset import BaseDataset


FrameSelector = Callable[[Any], Iterable[int] | Tensor]


@dataclass
class FrameSample:
    """
    A single frame extracted from an utterance-level sample.

    Attributes
    ----------
    utt_id:
        Identifier of the source utterance.
    utterance_index:
        Index of the source utterance in the wrapped dataset.
    frame_index:
        Index of this frame within the source utterance.
    resources:
        Frame-aligned resources indexed at `frame_index`.
    """

    utt_id: str
    utterance_index: int
    frame_index: int
    resources: Mapping[str, Any]


@dataclass
class FrameBatch:
    """A batch of independently sampled frames."""

    utt_ids: list[str]
    utterance_indices: Tensor
    frame_indices: Tensor
    resources: dict[str, Any]

    @classmethod
    def from_samples(cls, samples: list[FrameSample]) -> FrameBatch:
        if not samples:
            raise ValueError("Cannot collate an empty frame batch.")

        resource_names = tuple(samples[0].resources)

        for sample in samples[1:]:
            if tuple(sample.resources) != resource_names:
                raise ValueError("All frame samples in a batch must contain the same resources.")

        return cls(
            utt_ids=[sample.utt_id for sample in samples],
            utterance_indices=torch.tensor(
                [sample.utterance_index for sample in samples],
                dtype=torch.long,
            ),
            frame_indices=torch.tensor(
                [sample.frame_index for sample in samples],
                dtype=torch.long,
            ),
            resources={
                name: _collate_resource([sample.resources[name] for sample in samples]) for name in resource_names
            },
        )


def _collate_resource(values: list[Any]) -> Any:
    """
    Stack tensor-like resources when possible.

    Non-tensor resources are retained as a list.
    """
    try:
        return torch.stack([torch.as_tensor(value) for value in values])
    except (TypeError, ValueError, RuntimeError):
        return values


class FrameDataset(Dataset):
    """
    Wrap an utterance-level dataset and expose its frame-aligned resources as
    individual samples.

    The wrapped dataset may be constructed from a directory, glob pattern,
    manifest, or any other `BaseDataset` configuration. Rather than returning
    one item per utterance, this wrapper returns one item per selected frame.

    Each returned sample contains the same resource names configured on this
    wrapper, but every resource is indexed at one frame. For example, an
    utterance-level sample containing::

        sample.resources = {
            "wavlm": Tensor[T, 1024],
            "f0": Tensor[T],
            "voiced": Tensor[T],
        }

    becomes a frame-level sample containing::

        sample.resources = {
            "wavlm": Tensor[1024],
            "f0": Tensor[],
            "voiced": Tensor[],
        }

    `FrameDataset` does not assign semantic roles to resources. It does not
    distinguish model inputs from targets; the downstream training or
    evaluation module decides how each resource is used.

    Frame selection
    ---------------
    Exactly one frame-selection method must be provided:

    `length_resource`
        Includes every frame from every utterance. The named resource is used
        only to determine each utterance's frame count. This mode stores compact
        per-utterance counts and resolves global frame indices using cumulative
        lengths.

    `frame_selector`
        Selects arbitrary frame indices from each utterance. The selector
        receives a loaded utterance-level sample and returns its selected local
        frame indices. This mode stores an explicit
        `(utterance_index, frame_index)` mapping.

    Index caching
    -------------
    Building the frame mapping may require loading one or more resources from
    every utterance. When `index_path` is provided, the generated mapping is
    saved and reused on subsequent runs.

    The cached index depends on the ordering and contents of the wrapped
    dataset. Delete it or set `rebuild_index=True` whenever the wrapped dataset,
    its sorting, or the frame-selection logic changes.

    Parameters
    ----------
    dataset:
        The utterance-level dataset to wrap.
    resources:
        Names of the frame-aligned resources to include in each returned frame.
    length_resource:
        Resource whose first dimension determines the number of frames in an
        utterance. Mutually exclusive with `frame_selector`.
    frame_selector:
        Callable that returns the local frame indices to include for an
        utterance. Mutually exclusive with `length_resource`.
    index_path:
        Optional path at which to cache the generated frame mapping.
    rebuild_index:
        Rebuild the mapping even when `index_path` already exists.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        resources: Iterable[str],
        length_resource: str | None = None,
        frame_selector: FrameSelector | None = None,
        index_path: PathLike | None = None,
        rebuild_index: bool = False,
    ) -> None:
        # This validation belongs here, before attempting to load or build
        # either kind of frame index.
        if (length_resource is None) == (frame_selector is None):
            raise ValueError("Provide exactly one of `length_resource` or `frame_selector`.")

        self.dataset = dataset
        self.resources = tuple(resources)
        self.length_resource = length_resource
        self.frame_selector = frame_selector
        self.index_path = Path(index_path) if index_path is not None else None

        if not self.resources:
            raise ValueError("`resources` must contain at least one resource name.")

        if self.index_path is not None and self.index_path.exists() and not rebuild_index:
            index_data = torch.load(
                self.index_path,
                map_location="cpu",
                weights_only=True,
            )
        else:
            index_data = self._build_index()

            if self.index_path is not None:
                self.index_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                torch.save(index_data, self.index_path)

        if self.length_resource is not None:
            self._initialize_count_index(index_data)
            self.frame_index = None
        else:
            self._initialize_explicit_index(index_data)
            self.frame_counts = None
            self.cumulative_counts = None

    def _build_index(self) -> Tensor:
        if self.length_resource is not None:
            return self._build_frame_counts()

        return self._build_explicit_frame_index()

    def _build_frame_counts(self) -> Tensor:
        """Build one frame count per utterance."""
        counts: list[int] = []

        for utterance_index in range(len(self.dataset)):
            sample = self.dataset[utterance_index]

            try:
                resource = sample.resources[self.length_resource]
            except KeyError as error:
                raise KeyError(
                    f"Length resource {self.length_resource!r} is missing from utterance {sample.utt_id!r}."
                ) from error

            try:
                count = len(resource)
            except TypeError as error:
                raise TypeError(
                    f"Length resource {self.length_resource!r} for utterance {sample.utt_id!r} is not frame-indexable."
                ) from error

            counts.append(count)

        return torch.tensor(counts, dtype=torch.long)

    def _build_explicit_frame_index(self) -> Tensor:
        """
        Build `(utterance_index, frame_index)` pairs returned by the selector.
        """
        pairs: list[tuple[int, int]] = []

        assert self.frame_selector is not None

        for utterance_index in range(len(self.dataset)):
            sample = self.dataset[utterance_index]
            selected = self.frame_selector(sample)

            if isinstance(selected, Tensor):
                selected = selected.detach().cpu().flatten().tolist()

            for frame_index in selected:
                frame_index = int(frame_index)

                if frame_index < 0:
                    raise ValueError(
                        f"The frame selector returned negative frame index "
                        f"{frame_index} for utterance {sample.utt_id!r}."
                    )

                pairs.append((utterance_index, frame_index))

        if not pairs:
            return torch.empty((0, 2), dtype=torch.long)

        return torch.tensor(pairs, dtype=torch.long)

    def _initialize_count_index(self, index_data: Tensor) -> None:
        if index_data.ndim != 1:
            raise ValueError(
                "A cached `length_resource` index must be a one-dimensional "
                "tensor containing one frame count per utterance."
            )

        if len(index_data) != len(self.dataset):
            raise ValueError(
                "The cached frame counts do not match the wrapped dataset: "
                f"found {len(index_data)} counts for "
                f"{len(self.dataset)} utterances. Rebuild the index."
            )

        if torch.any(index_data < 0):
            raise ValueError("Cached frame counts cannot be negative.")

        self.frame_counts = index_data.long()
        self.cumulative_counts = torch.cumsum(
            self.frame_counts,
            dim=0,
        )
        self._cumulative_counts_list = self.cumulative_counts.tolist()

    def _initialize_explicit_index(self, index_data: Tensor) -> None:
        if index_data.ndim != 2 or index_data.shape[1] != 2:
            raise ValueError(
                "A cached selector index must have shape [N, 2], containing `(utterance_index, frame_index)` pairs."
            )

        self.frame_index = index_data.long()

        if len(self.frame_index) == 0:
            return

        utterance_indices = self.frame_index[:, 0]

        if torch.any(utterance_indices < 0) or torch.any(utterance_indices >= len(self.dataset)):
            raise ValueError(
                "The cached frame index refers to utterances outside the wrapped dataset. Rebuild the index."
            )

        if torch.any(self.frame_index[:, 1] < 0):
            raise ValueError("The cached frame index contains negative frame indices.")

    def __len__(self) -> int:
        if self.length_resource is not None:
            if len(self.cumulative_counts) == 0:
                return 0

            return int(self.cumulative_counts[-1])

        return len(self.frame_index)

    def _resolve_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(index)

        if self.length_resource is None:
            utterance_index, frame_index = self.frame_index[index].tolist()
            return utterance_index, frame_index

        # bisect_right expects an ordinary sequence. Tensor.tolist() is okay,
        # although storing this list once avoids recreating it per item.
        cumulative_counts = self.cumulative_counts.tolist()

        utterance_index = bisect_right(
            self._cumulative_counts_list,
            index,
        )

        previous_total = cumulative_counts[utterance_index - 1] if utterance_index > 0 else 0

        frame_index = index - previous_total
        return utterance_index, frame_index

    def __getitem__(self, index: int) -> FrameSample:
        utterance_index, frame_index = self._resolve_index(index)
        sample = self.dataset[utterance_index]

        frame_resources: dict[str, Any] = {}

        for name in self.resources:
            try:
                resource = sample.resources[name]
            except KeyError as error:
                raise KeyError(f"Resource {name!r} is missing from utterance {sample.utt_id!r}.") from error

            try:
                resource_length = len(resource)
            except TypeError as error:
                raise TypeError(f"Resource {name!r} for utterance {sample.utt_id!r} is not frame-indexable.") from error

            if frame_index >= resource_length:
                raise IndexError(
                    f"Frame {frame_index} is outside resource {name!r} for "
                    f"utterance {sample.utt_id!r}, which contains "
                    f"{resource_length} frames."
                )

            frame_resources[name] = resource[frame_index]

        return FrameSample(
            utt_id=sample.utt_id,
            utterance_index=utterance_index,
            frame_index=frame_index,
            resources=frame_resources,
        )

    def collate_fn(
        self,
        samples: list[FrameSample],
    ) -> FrameBatch:
        return FrameBatch.from_samples(samples)

    def make_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        **kwargs: Any,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self.collate_fn,
            **kwargs,
        )
