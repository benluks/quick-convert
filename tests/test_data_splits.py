import pandas as pd

from quick_convert.data.splits import (
    split_manifest_group_disjoint,
    split_manifest_within_groups,
)


def test_split_manifest_partitions_rows_within_each_group():
    manifest = pd.DataFrame(
        {
            "utt_id": [f"a-{i}" for i in range(10)] + [f"b-{i}" for i in range(5)] + ["c-0"],
            "spkid": ["a"] * 10 + ["b"] * 5 + ["c"],
        }
    )

    train, valid = split_manifest_within_groups(manifest, valid_fraction=0.2, seed=7)

    assert set(train["spkid"]) == {"a", "b", "c"}
    assert set(valid["spkid"]) == {"a", "b"}
    assert set(train["utt_id"]).isdisjoint(valid["utt_id"])
    assert set(train["utt_id"]) | set(valid["utt_id"]) == set(manifest["utt_id"])
    assert len(valid[valid["spkid"] == "a"]) == 2
    assert len(valid[valid["spkid"] == "b"]) == 1


def test_group_disjoint_split_assigns_each_speaker_to_one_partition():
    manifest = pd.DataFrame(
        {
            "utt_id": [
                "a-picnic",
                "a-rainbow",
                "b-picnic",
                "b-rainbow",
                "c-picnic",
                "c-rainbow",
                "d-picnic",
                "d-rainbow",
            ],
            "spkid": ["a", "a", "b", "b", "c", "c", "d", "d"],
        }
    )

    train, valid = split_manifest_group_disjoint(
        manifest,
        valid_fraction=0.25,
        seed=115,
    )

    assert set(train["spkid"]).isdisjoint(valid["spkid"])
    assert set(train["utt_id"]) | set(valid["utt_id"]) == set(manifest["utt_id"])
    assert len(set(valid["spkid"])) == 1
    assert len(valid) == 2


def test_group_disjoint_split_is_deterministic():
    manifest = pd.DataFrame(
        {
            "utt_id": [f"{speaker}-{index}" for speaker in "abcdef" for index in range(2)],
            "spkid": [speaker for speaker in "abcdef" for _ in range(2)],
        }
    )

    first = split_manifest_group_disjoint(manifest, valid_fraction=0.33, seed=7)
    second = split_manifest_group_disjoint(manifest, valid_fraction=0.33, seed=7)

    assert first[0]["utt_id"].tolist() == second[0]["utt_id"].tolist()
    assert first[1]["utt_id"].tolist() == second[1]["utt_id"].tolist()
