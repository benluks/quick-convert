"""Dataset and manifest partitioning helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import pandas as pd


def split_manifest_within_groups(
    manifest: pd.DataFrame,
    group_col: str = "spkid",
    valid_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each group's rows between training and validation.

    Every group with more than one row contributes at least one validation
    row. Consequently, multi-row groups normally occur in both partitions;
    this function does not create group-disjoint splits.
    """
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError("Manifest splitting requires the `manifests` extra.") from error

    train_parts = []
    valid_parts = []

    for _, group in manifest.groupby(group_col):
        valid = group.sample(frac=valid_fraction, random_state=seed)

        if len(valid) == 0 and len(group) > 1:
            valid = group.sample(n=1, random_state=seed)

        train_parts.append(group.drop(valid.index))
        valid_parts.append(valid)

    return (
        pd.concat(train_parts).sample(frac=1, random_state=seed),
        pd.concat(valid_parts).sample(frac=1, random_state=seed),
    )
