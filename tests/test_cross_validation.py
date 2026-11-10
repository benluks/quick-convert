import pytest

from quick_convert.data.cross_validation import assign_folds, fold_indices


def test_grouped_folds_are_disjoint_exhaustive_and_order_independent():
    group_ids = ["a", "b", "a", "c", "d", "c", "e", "f", "a"]
    assignments = assign_folds(group_ids, n_splits=3, seed=17)
    assert assignments == assign_folds(reversed(group_ids), n_splits=3, seed=17)

    held_out_indices = []
    for fold in range(3):
        train, held_out = fold_indices(group_ids, assignments, fold=fold)
        assert sorted(train + held_out) == list(range(len(group_ids)))
        assert {group_ids[index] for index in train}.isdisjoint(group_ids[index] for index in held_out)
        assert len(held_out) == 3
        held_out_indices.extend(held_out)
    assert sorted(held_out_indices) == list(range(len(group_ids)))


def test_rejects_invalid_groups_and_folds():
    with pytest.raises(ValueError, match="distinct groups"):
        assign_folds(["a", "a"], n_splits=2)
    with pytest.raises(ValueError, match="nonempty strings"):
        assign_folds(["a", " "], n_splits=2)
    with pytest.raises(ValueError, match="Missing fold assignment"):
        fold_indices(["a", "b"], {"a": 0}, fold=0)
    with pytest.raises(ValueError, match="both training and held-out"):
        fold_indices(["a", "b"], {"a": 0, "b": 1}, fold=2)
