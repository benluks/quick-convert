"""Dataset and manifest partitioning helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import pandas as pd


def _validate_split_inputs(
    manifest: pd.DataFrame,
    group_col: str,
    valid_fraction: float,
) -> None:
    if group_col not in manifest.columns:
        raise ValueError(f"Manifest has no group column {group_col!r}.")
    if manifest.empty:
        raise ValueError("Cannot split an empty manifest.")
    if manifest[group_col].isna().any():
        raise ValueError(f"Manifest group column {group_col!r} contains missing values.")
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1.")


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
        raise ImportError("Manifest splitting requires the manifests extra.") from error

    _validate_split_inputs(manifest, group_col, valid_fraction)

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


def split_manifest_group_disjoint(
    manifest: pd.DataFrame,
    group_col: str = "spkid",
    valid_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign complete groups to either training or validation.

    The number of validation groups is rounded from valid_fraction and
    clamped so that both partitions contain at least one group.
    """
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError("Manifest splitting requires the manifests extra.") from error

    _validate_split_inputs(manifest, group_col, valid_fraction)

    groups = manifest[group_col].drop_duplicates()
    if len(groups) < 2:
        raise ValueError("A group-disjoint split requires at least two groups.")

    valid_count = round(len(groups) * valid_fraction)
    valid_count = min(max(valid_count, 1), len(groups) - 1)
    valid_groups = set(groups.sample(n=valid_count, random_state=seed))

    valid_mask = manifest[group_col].isin(valid_groups)
    train = manifest.loc[~valid_mask]
    valid = manifest.loc[valid_mask]

    return (
        train.sample(frac=1, random_state=seed),
        valid.sample(frac=1, random_state=seed),
    )
