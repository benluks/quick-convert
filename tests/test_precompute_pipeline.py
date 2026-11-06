import json

import pytest
import torch

from quick_convert.data import AudioSample, BaseDataset
from quick_convert.pipelines.precompute_features import PrecomputeFeaturesPipeline


class TrackingDataset(BaseDataset):
    def __init__(self, rows):
        super().__init__(rows=rows)
        self.loaded = []

    def __getitem__(self, index):
        self.loaded.append(index)
        return super().__getitem__(index)


class FixedExtractor:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def extract_batch(self, batch):
        self.calls.append(batch.utt_ids)
        return self.outputs


def make_dataset(tmp_path):
    return TrackingDataset(
        [
            AudioSample(utt_id="first", path=tmp_path / "a.flac", split="train"),
            AudioSample(utt_id="second", path=tmp_path / "b.flac", split="validation"),
        ]
    )


def test_precompute_writes_one_feature_and_manifest_row_per_sample(tmp_path):
    dataset = make_dataset(tmp_path)
    outputs = [torch.tensor([1.0]), torch.tensor([2.0])]
    output_dir = tmp_path / "features"

    PrecomputeFeaturesPipeline(
        dataset=dataset,
        batch_size=2,
        extractor=FixedExtractor(outputs),
        out_dir=output_dir,
    ).run()

    rows = [json.loads(line) for line in (output_dir / "manifest.jsonl").read_text().splitlines()]

    assert len(rows) == len(dataset)
    for sample, expected, row in zip(dataset.rows, outputs, rows, strict=True):
        assert row["utt_id"] == sample.utt_id
        assert row["path"] == str(sample.path)
        assert row["split"] == sample.split
        assert row["feature_path"].endswith(f"/{sample.split}/{sample.utt_id}.pt")
        assert torch.equal(torch.load(row["feature_path"], weights_only=True), expected)


def test_precompute_rejects_output_count_mismatch(tmp_path):
    dataset = make_dataset(tmp_path)
    pipeline = PrecomputeFeaturesPipeline(
        dataset=dataset,
        batch_size=2,
        extractor=FixedExtractor([torch.tensor([1.0])]),
        out_dir=tmp_path / "features",
    )

    with pytest.raises(ValueError, match="returned 1 outputs for batch of size 2"):
        pipeline.run()


def test_precompute_resume_keeps_existing_features_in_manifest(tmp_path):
    dataset = make_dataset(tmp_path)
    output_dir = tmp_path / "features"
    extractor = FixedExtractor([torch.tensor([10.0]), torch.tensor([20.0])])
    pipeline = PrecomputeFeaturesPipeline(
        dataset, extractor, output_dir, batch_size=2, skip_existing=True, cache_key="inputs-v1"
    )
    pipeline.run()
    manifest = (output_dir / "manifest.jsonl").read_text()
    (output_dir / "validation" / "second.pt").unlink()
    dataset.loaded.clear()
    extractor.calls.clear()
    extractor.outputs = [torch.tensor([20.0])]
    pipeline.run()

    assert dataset.loaded == [1]
    assert extractor.calls == [["second"]]
    assert (output_dir / "manifest.jsonl").read_text() == manifest
    assert torch.equal(torch.load(output_dir / "train" / "first.pt", weights_only=True), torch.tensor([10.0]))

    dataset.loaded.clear()
    extractor.calls.clear()
    pipeline.run()
    assert dataset.loaded == extractor.calls == []


def test_precompute_interruption_preserves_manifest_and_hides_partial_feature(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path)
    output_dir = tmp_path / "features"
    pipeline = PrecomputeFeaturesPipeline(
        dataset,
        FixedExtractor([torch.ones(1), torch.ones(1)]),
        output_dir,
        batch_size=2,
        skip_existing=True,
        cache_key="inputs-v1",
    )
    pipeline.run()
    manifest_path = output_dir / "manifest.jsonl"
    original = manifest_path.read_bytes()
    missing_path = output_dir / "validation" / "second.pt"
    missing_path.unlink()
    pipeline.extractor.outputs = [torch.ones(1)]

    def fail(_output, stream):
        stream.write(b"incomplete")
        raise RuntimeError("interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", fail)
        with pytest.raises(RuntimeError, match="interrupted"):
            pipeline.run()
    assert manifest_path.read_bytes() == original
    assert not missing_path.exists()
    assert list(missing_path.parent.iterdir()) == []
    pipeline.run()
    assert torch.equal(torch.load(missing_path, weights_only=True), torch.ones(1))


def test_precompute_rejects_changed_config_before_loading_samples(tmp_path):
    dataset = make_dataset(tmp_path)
    pipeline = PrecomputeFeaturesPipeline(
        dataset,
        FixedExtractor([torch.ones(1), torch.ones(1)]),
        tmp_path / "features",
        batch_size=2,
        skip_existing=True,
    )
    pipeline.write_config("pipeline: {extractor: {layer: 6}, dataset: {target_sr: 16000}, batch_size: 2}")
    pipeline.run()
    pipeline.write_config("pipeline: {extractor: {layer: 6}, dataset: {target_sr: 16000}, batch_size: 1}")
    pipeline.run()
    dataset.loaded.clear()
    pipeline.write_config("pipeline: {extractor: {layer: 7}, dataset: {target_sr: 16000}}")
    with pytest.raises(ValueError, match="identity changed"):
        pipeline.run()
    assert dataset.loaded == []


def test_precompute_rejects_legacy_cache_and_duplicate_destinations(tmp_path):
    dataset = make_dataset(tmp_path)
    output_dir = tmp_path / "features"
    old_path = output_dir / "train" / "first.pt"
    old_path.parent.mkdir(parents=True)
    old_path.touch()
    pipeline = PrecomputeFeaturesPipeline(
        dataset, FixedExtractor([]), output_dir, skip_existing=True, cache_key="inputs-v1"
    )
    with pytest.raises(ValueError, match="no cache identity"):
        pipeline.run()
    dataset.rows.append(dataset.rows[0])
    with pytest.raises(ValueError, match="Duplicate feature output path"):
        pipeline.run()
    assert dataset.loaded == []
    assert not (output_dir / "provenance.json").exists()


def test_cached_content_preserves_online_representation_contract(tmp_path):
    from types import SimpleNamespace

    from quick_convert.components.feature_extractors.content import ContentFeatureExtractor
    from quick_convert.components.layers import LayerWeightedSum
    from quick_convert.components.ssl.base import ContentFeatures
    from quick_convert.data.resources import PathResourceProvider
    from quick_convert.systems.conversion import SSLReconstructionConversionSystem

    class Encoder(torch.nn.Module):
        def forward(self, batch):
            return ContentFeatures(
                values=torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
                * torch.tensor([[1, 0, 0], [1, 1, 1]])[..., None, None],
                lengths=torch.tensor([1, 3]),
                feature_dim=2,
                representation_type="continuous",
                temporal_granularity="frame",
                backend="test",
                model_name="layered",
                layer="all",
                frame_hz=25.0,
            )

    dataset = make_dataset(tmp_path)
    encoder = Encoder()
    decoder = torch.nn.Identity()
    decoder.flow = SimpleNamespace(input_frame_rate=25)
    system = SSLReconstructionConversionSystem(
        decoder, feature_transform=LayerWeightedSum(2), online_encoders={"content": encoder}
    )
    online_batch = next(iter(dataset.make_dataloader(batch_size=2)))
    online = system.get_resource(online_batch, "content")
    out_dir = tmp_path / "features"
    PrecomputeFeaturesPipeline(dataset, ContentFeatureExtractor(encoder), out_dir, batch_size=2).run()

    cached_dataset = BaseDataset(
        rows=dataset.rows,
        load=["content"],
        resource_providers=[
            PathResourceProvider("content", f"{out_dir}/{{sample.split}}/{{sample.utt_id}}.pt", "torch_tensor")
        ],
    )
    cached_batch = next(iter(cached_dataset.make_dataloader(batch_size=2)))
    cached = system.get_resource(cached_batch, "content")
    torch.testing.assert_close(cached.values, online.values)
    torch.testing.assert_close(cached.lengths, online.lengths)
    assert cached.metadata == online.metadata
    assert cached_batch[0].resources["content"].value.shape == (1, 2, 2)

    def adapt(resource):
        return system.encode_features(resource.values, resource.lengths, metadata=resource.metadata)[:2]

    torch.testing.assert_close(adapt(cached), adapt(online))
    decoder.flow.input_frame_rate = 50
    for resource in (online, cached):
        with pytest.raises(ValueError, match="frame rate"):
            adapt(resource)

    first, second = [out_dir / sample.split / f"{sample.utt_id}.pt" for sample in dataset.rows]
    torch.save(torch.ones(3, 2), first)
    with pytest.raises(ValueError, match="inconsistent representation metadata"):
        next(iter(cached_dataset.make_dataloader(batch_size=2)))
    torch.save(torch.ones(3, 2), second)
    legacy = system.get_resource(next(iter(cached_dataset.make_dataloader(batch_size=2))), "content")
    assert legacy.metadata is None
    torch.testing.assert_close(legacy.values, torch.ones(2, 3, 2))
