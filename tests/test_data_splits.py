import pandas as pd

from quick_convert.data.splits import split_manifest_within_groups


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
