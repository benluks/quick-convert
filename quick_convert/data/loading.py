# quick_convert/data/loading.py

from __future__ import annotations

from collections.abc import Iterable
from os import PathLike

from hydra.utils import instantiate

from quick_convert.utils.config import compose_component

from .base_dataset import BaseDataset
from .resources import BaseResourceProvider


def load_dataset(
    name: str | None = None,
    *,
    root: str | PathLike | None = None,
    splits: Iterable[str] | None = None,
    additional_resource_providers: Iterable[BaseResourceProvider] | None = None,
    **overrides,
) -> BaseDataset:
    values = dict(overrides)

    if root is not None:
        values["root"] = root

    if splits is not None:
        values["splits"] = list(splits)

    if name is None:
        if root is None:
            raise ValueError("Either `name` or `root` must be provided.")

        dataset = BaseDataset(**values)
    else:
        cfg = compose_component(
            "dataset",
            name,
            overrides=values,
        )
        dataset = instantiate(cfg)

    if not isinstance(dataset, BaseDataset):
        raise TypeError(f"Expected BaseDataset, got {type(dataset).__name__}.")

    if additional_resource_providers:
        dataset.resource_providers.extend(additional_resource_providers)

    return dataset
