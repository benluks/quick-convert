"""Small contract and persistence helpers for dataset-fitted transformations."""

from collections.abc import Iterable, Mapping
from typing import Any, Protocol, Self, TypeVar

import torch


InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)


class FittedTransform(Protocol[InputT, OutputT]):
    """Fit training records, transform a record, and expose portable learned state."""

    def fit(self, records: Iterable[InputT]) -> Self: ...
    def transform(self, record: InputT) -> OutputT: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> Self: ...


def save_fitted_state(transform: FittedTransform, path):
    """Persist plain fitted state; the workflow owns the destination directory."""
    torch.save(dict(transform.state_dict()), path)


def load_fitted_state(transform: FittedTransform, path):
    """Restore state into a configured transform without unpickling an object."""
    return transform.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
