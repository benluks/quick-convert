"""Group-disjoint fold assignment and selection for indexed datasets."""

from collections import Counter
from collections.abc import Iterable, Mapping
from random import Random


def assign_folds(group_ids: Iterable[str], *, n_splits: int, seed: int = 0) -> dict[str, int]:
    """Assign whole groups to zero-based folds, balancing the number of items.

    Supply one nonempty group ID per item. Larger groups are placed first in
    the fold with the fewest items. The seed breaks ties between groups of
    equal size, independently of input order. The returned mapping can be
    reused across dataset views or serialized as JSON.
    """
    group_sizes = Counter(group_ids)
    if any(not isinstance(group_id, str) or not group_id.strip() for group_id in group_sizes):
        raise ValueError("Group IDs must be nonempty strings.")
    if not 2 <= n_splits <= len(group_sizes):
        raise ValueError("n_splits must be between 2 and the number of distinct groups.")

    groups = sorted(group_sizes)
    Random(seed).shuffle(groups)
    groups.sort(key=group_sizes.__getitem__, reverse=True)

    fold_sizes = [0] * n_splits
    assignments = {}
    for group_id in groups:
        fold = min(range(n_splits), key=fold_sizes.__getitem__)
        assignments[group_id] = fold
        fold_sizes[fold] += group_sizes[group_id]
    return assignments


def fold_indices(group_ids: Iterable[str], assignments: Mapping[str, int], *, fold: int) -> tuple[list[int], list[int]]:
    """Return training and held-out indices in the order of the input items."""
    train, held_out = [], []
    for index, group_id in enumerate(group_ids):
        if group_id not in assignments:
            raise ValueError(f"Missing fold assignment for group {group_id!r}.")
        indices = held_out if assignments[group_id] == fold else train
        indices.append(index)
    if not train or not held_out:
        raise ValueError(f"Fold {fold} must have both training and held-out items.")
    return train, held_out
