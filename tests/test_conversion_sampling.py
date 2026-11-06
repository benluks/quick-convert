import json
from pathlib import Path

import pytest
import torch
import yaml
from test_grouped_conversion import System, sample

from quick_convert.data import BaseDataset
from quick_convert.data.sampling import shuffled_sample_indices
from quick_convert.pipelines.conversion import SpeechConversionPipeline
from quick_convert.systems.conversion.targets import ReferenceSpeech


def test_sampling_is_a_reproducible_shuffled_prefix_without_global_rng_changes():
    state = torch.get_rng_state().clone()
    small = shuffled_sample_indices(30, 5, 42)
    assert small == shuffled_sample_indices(30, 5, 42)
    assert small == shuffled_sample_indices(30, 12, 42)[:5]
    assert small != list(range(5))
    assert small != shuffled_sample_indices(30, 5, 43)
    assert len(set(small)) == 5
    assert sorted(shuffled_sample_indices(3, 20, 42)) == [0, 1, 2]
    assert shuffled_sample_indices(0, 10, 42) == []
    assert shuffled_sample_indices(10, None) is None
    torch.testing.assert_close(state, torch.get_rng_state())


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limits_fail_without_dataset_loading(limit):
    with pytest.raises(ValueError, match="max_samples"):
        shuffled_sample_indices(10, limit)


def test_a_seed_alone_is_not_silently_ignored():
    with pytest.raises(ValueError, match="requires max_samples"):
        shuffled_sample_indices(10, None, 42)


def test_pipeline_limits_before_loading_and_handles_partial_batches_and_resume(tmp_path):
    class CountingDataset(BaseDataset):
        def __getitem__(self, index):
            self.read.append(index)
            return super().__getitem__(index)

    dataset = CountingDataset(rows=[sample(f"u-{i}", "source") for i in range(30)])
    dataset.read = []
    target = ReferenceSpeech([Path(f"/target/ref-{i}.wav") for i in range(20)])
    pipeline = SpeechConversionPipeline(
        System(), dataset, target, tmp_path, max_samples=5, sample_seed=42, batch_size=4
    )
    pipeline.run()
    assert len(pipeline.system.prepared[0]) == 20
    expected = shuffled_sample_indices(30, 5, 42)
    assert dataset.read == expected
    rows = [json.loads(line) for line in (tmp_path / "conversions.jsonl").read_text().splitlines()]
    assert [row["utt_id"] for row in rows] == [f"u-{i}" for i in expected]
    assert len(list((tmp_path / "test").glob("*.wav"))) == 5
    run = yaml.safe_load((tmp_path / "run.yaml").read_text())
    assert run["metadata"]["source_sampling"] == {"max_samples": 5, "seed": 42, "population": 30, "selected": 5}
    assert rows[0]["source_sampling"] == run["metadata"]["source_sampling"]

    resumed = System()
    SpeechConversionPipeline(resumed, dataset, target, tmp_path, max_samples=5, sample_seed=42, batch_size=2).run()
    assert not resumed.converted and not resumed.prepared
    assert len((tmp_path / "conversions.jsonl").read_text().splitlines()) == 5

    SpeechConversionPipeline(System(), dataset, target, tmp_path, max_samples=8, sample_seed=42).run()
    assert len(list((tmp_path / "test").glob("*.wav"))) == 8
