# quick_convert/data/loading.py

from __future__ import annotations

from collections.abc import Iterable
from os import PathLike

from hydra.utils import instantiate

from quick_convert.utils.config import compose_component

from .base_dataset import BaseDataset
from .resources import BaseResourceProvider


def load_dataset(
    name: str,
    *,
    root: str | PathLike | None = None,
    splits: Iterable[str] | None = None,
    additional_resource_providers: Iterable[BaseResourceProvider] | None = None,
    **overrides,
) -> BaseDataset:
    """Instantiate a packaged quick-convert dataset recipe.

    Args:
        name:
            Name of a dataset config in the ``dataset`` config group.
        root:
            Optional local dataset root overriding the packaged recipe.
        splits:
            Optional dataset splits overriding the packaged recipe.
        additional_resource_providers:
            Providers to append to those already defined by the dataset recipe.
        **overrides:
            Additional dataset constructor/config overrides.

    Example:
        ::

            dataset = load_dataset(
                "librispeech",
                root="/data/LibriSpeech",
                splits=["train-clean-100"],
                load=["audio"],
            )
    """

    values = dict(overrides)

    if root is not None:
        values["root"] = root

    if splits is not None:
        values["splits"] = list(splits)

    cfg = compose_component(
        "dataset",
        name,
        overrides=values,
    )

    dataset = instantiate(cfg)

    if not isinstance(dataset, BaseDataset):
        raise TypeError(f"Dataset recipe {name!r} produced {type(dataset).__name__}, expected BaseDataset.")

    if additional_resource_providers:
        dataset.resource_providers.extend(additional_resource_providers)

    return dataset
