from collections.abc import Iterable
from typing import Any

import torch

from quick_convert.utils.paths import TemplateFormatter


class Indexer:
    """
    Arbitrary index over hashable values discovered from rows/samples.
    Useful for speaker ids, languages, labels, etc.
    """

    def __init__(self, template: str, sort: bool = True, state_path=None):
        self.template = template
        self.sort = sort
        self.value_to_idx: dict[Any, int] = {}
        self.idx_to_value: dict[int, Any] = {}
        self.state_path = state_path
        if state_path is not None:
            from quick_convert.components.fitting import load_fitted_state

            load_fitted_state(self, state_path)

    def resolve(self, row):
        return TemplateFormatter.format_str(
            self.template,
            sample=row,
            row=row,
            path=row.path,
            resources=row.resources,
        )

    def fit(self, rows: Iterable):
        values = [self.resolve(row) for row in rows]

        labels = sorted(set(values)) if self.sort else list(dict.fromkeys(values))

        self.value_to_idx = {value: idx for idx, value in enumerate(labels)}
        self.idx_to_value = {idx: value for value, idx in self.value_to_idx.items()}

        return self

    def encode(self, value: Any) -> int:
        return self.value_to_idx[value]

    def transform(self, row) -> int:
        """Apply the fitted index to a row, using the same selection as fitting."""
        return self.encode(self.resolve(row))

    def state_dict(self):
        """Plain mapping state usable by fitting workflows and checkpoints."""
        return {
            "version": 1,
            "config": {"template": self.template, "sort": self.sort},
            "value_to_idx": dict(self.value_to_idx),
        }

    def load_state_dict(self, state):
        if state["version"] != 1 or state["config"] != {"template": self.template, "sort": self.sort}:
            raise ValueError("Indexer state does not match the configured index.")
        mapping = dict(state["value_to_idx"])
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError("Indexer labels must form a contiguous integer mapping.")
        self.value_to_idx = mapping
        self.idx_to_value = {index: value for value, index in mapping.items()}
        return self

    def encode_many(self, values: Iterable[Any]) -> list[int]:
        return [self.encode(value) for value in values]

    def encode_tensor(
        self,
        values: Iterable[Any],
        device=None,
        dtype=torch.long,
    ) -> torch.Tensor:
        return torch.tensor(
            self.encode_many(values),
            device=device,
            dtype=dtype,
        )

    def decode(self, idx: int) -> Any:
        return self.idx_to_value[int(idx)]

    def decode_many(self, indices: Iterable[int]) -> list[Any]:
        return [self.decode(idx) for idx in indices]

    def __len__(self):
        return len(self.value_to_idx)

    def __repr__(self):
        if self.idx_to_value:
            return f"{type(self).__name__}(n={len(self)}, values={self.idx_to_value})"
        return f"{type(self).__name__}(template={self.template!r}, unfitted)"
