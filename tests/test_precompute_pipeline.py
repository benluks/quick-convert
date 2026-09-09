import json

import pytest
import torch

from quick_convert.data import AudioBatch, AudioSample
from quick_convert.pipelines.precompute_features import PrecomputeFeaturesPipeline


class BatchDataset:
    def __init__(self, batch):
        self.batch = batch

    def make_dataloader(self, **kwargs):
        return [self.batch]


class FixedExtractor:
    def __init__(self, outputs):
        self.outputs = outputs

    def extract_batch(self, batch):
        return self.outputs


def make_batch(tmp_path):
    return AudioBatch.from_samples(
        [
            AudioSample(utt_id="first", path=tmp_path / "a.flac", split="train"),
            AudioSample(utt_id="second", path=tmp_path / "b.flac", split="validation"),
        ]
    )


def test_precompute_writes_one_feature_and_manifest_row_per_sample(tmp_path):
    batch = make_batch(tmp_path)
    outputs = [torch.tensor([1.0]), torch.tensor([2.0])]
    output_dir = tmp_path / "features"

    PrecomputeFeaturesPipeline(
        dataset=BatchDataset(batch),
        extractor=FixedExtractor(outputs),
        out_dir=output_dir,
    ).run()

    rows = [json.loads(line) for line in (output_dir / "manifest.jsonl").read_text().splitlines()]

    assert len(rows) == len(batch)
    for sample, expected, row in zip(batch, outputs, rows, strict=True):
        assert row["utt_id"] == sample.utt_id
        assert row["path"] == str(sample.path)
        assert row["split"] == sample.split
        assert row["feature_path"].endswith(f"/{sample.split}/{sample.utt_id}.pt")
        assert torch.equal(torch.load(row["feature_path"], weights_only=True), expected)


def test_precompute_rejects_output_count_mismatch(tmp_path):
    batch = make_batch(tmp_path)
    pipeline = PrecomputeFeaturesPipeline(
        dataset=BatchDataset(batch),
        extractor=FixedExtractor([torch.tensor([1.0])]),
        out_dir=tmp_path / "features",
    )

    with pytest.raises(ValueError, match="returned 1 outputs for batch of size 2"):
        pipeline.run()
