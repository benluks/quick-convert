import json
from pathlib import Path

import pytest
import torch

from quick_convert.data import AudioSample, BaseDataset, GeneratedAudio
from quick_convert.data.resources import ResourceCollection, ResourceRef
from quick_convert.pipelines.conversion import SpeechConversionPipeline
from quick_convert.pipelines.targets import DatasetReferenceTargets
from quick_convert.systems.anonymization.targets import PreparedTarget


def sample(utt_id, label, split="test"):
    return AudioSample(
        utt_id=utt_id,
        path=Path(f"/audio/{utt_id}.wav"),
        split=split,
        resources=ResourceCollection.from_refs([ResourceRef("identity", "text", value=label)]),
        waveform=torch.arange(3, dtype=torch.float32),
        sample_rate=16000,
    )


def targets(**kwargs):
    references = BaseDataset(
        rows=[sample("ref-a", "a"), sample("ref-a2", "a"), sample("ref-b", "b"), sample("ref-c", "c")]
    )
    return DatasetReferenceTargets(references, resource="identity", **kwargs)


class System:
    def __init__(self):
        self.prepared = []
        self.converted = []

    def prepare_target(self, request):
        self.prepared.append(request.paths)
        return PreparedTarget(request.paths, "test-space", "test", references=request.paths)

    def convert_batch(self, batch, *, target):
        self.converted.append((batch.utt_ids[0], target.references))
        return GeneratedAudio(batch.waveforms.clone(), batch.lengths.clone(), 16000)


def test_grouping_fixed_and_random_selection_order_independence():
    fixed = targets(value="a")
    assert len(fixed.groups["a"].paths) == 2
    assert fixed.select(sample("utt", "b")) == "a"
    random = targets(mode="random", seed=12, exclude_source=True)
    sources = [sample(f"u-{i}", "a") for i in range(30)]
    assignments = {s.utt_id: random.select(s) for s in sources}
    assert set(assignments.values()) == {"b", "c"}
    assert assignments == {s.utt_id: random.select(s) for s in reversed(sources)}
    other_seed = targets(mode="random", seed=13, exclude_source=True)
    assert assignments != {s.utt_id: other_seed.select(s) for s in sources}


def test_selection_errors():
    with pytest.raises(ValueError, match="requires a group"):
        targets()
    with pytest.raises(ValueError, match="does not accept"):
        targets(mode="random", value="a")
    with pytest.raises(ValueError, match="No recordings"):
        targets(value="absent")
    with pytest.raises(ValueError, match="excluded source"):
        targets(value="a", exclude_source=True).select(sample("u", "a"))
    only = DatasetReferenceTargets(
        BaseDataset(rows=[sample("ref", "a")]), resource="identity", mode="random", exclude_source=True
    )
    with pytest.raises(ValueError, match="No eligible"):
        only.select(sample("u", "a"))
    with pytest.raises(ValueError, match="lacks speaker resource"):
        targets(mode="random", exclude_source=True, source_resource="missing").select(sample("u", "a"))


def test_grouped_conversion_batch_independence_cache_resume_and_logging(tmp_path, monkeypatch, caplog, capsys):
    monkeypatch.setattr(
        "torchaudio.save",
        lambda path, waveform, sr: Path(path).write_bytes(b"audio"),
    )
    sources = [sample(f"u-{i}", "a") for i in range(12)]
    selector = targets(mode="random", seed=3, exclude_source=True)
    system = System()
    caplog.set_level("INFO")
    SpeechConversionPipeline(system, BaseDataset(rows=sources), selector, tmp_path / "first", batch_size=4).run()
    assert len(system.prepared) == 2
    records = [json.loads(line) for line in (tmp_path / "first/target_assignments.jsonl").read_text().splitlines()]
    assert len(records) == len(sources)
    assert {r["target_value"] for r in records} == {"b", "c"}
    assert all(r["references"] and r["catalogue_sha256"] for r in records)
    assert "Conversion complete: 12 converted" in caplog.text
    assert "Converting speech" in capsys.readouterr().err
    resumed = System()
    SpeechConversionPipeline(resumed, BaseDataset(rows=sources), selector, tmp_path / "first", batch_size=1).run()
    assert not resumed.prepared and not resumed.converted
    reordered = System()
    SpeechConversionPipeline(
        reordered, BaseDataset(rows=list(reversed(sources))), selector, tmp_path / "second", batch_size=3
    ).run()
    assert dict(system.converted) == dict(reordered.converted)
    changed = targets(mode="random", seed=4, exclude_source=True)
    with pytest.raises(ValueError, match="assignment differs"):
        SpeechConversionPipeline(System(), BaseDataset(rows=sources), changed, tmp_path / "first").run()


def test_lru_cache_eviction_and_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "torchaudio.save",
        lambda path, waveform, sr: Path(path).write_bytes(b"audio"),
    )
    selector = targets(mode="random", seed=3)
    sources = [sample(f"u-{i}", "a") for i in range(20)]
    system = System()
    SpeechConversionPipeline(system, BaseDataset(rows=sources), selector, tmp_path, max_prepared_targets=1).run()
    labels = [selector.select(s) for s in sources]
    expected = 1 + sum(a != b for a, b in zip(labels, labels[1:], strict=False))
    assert len(system.prepared) == expected
    replacement = System()
    SpeechConversionPipeline(replacement, BaseDataset(rows=sources), targets(value="b"), tmp_path, overwrite=True).run()
    assert len(replacement.prepared) == 1
    assert len(replacement.converted) == len(sources)


def test_pipeline_forwards_per_call_cfg(tmp_path):
    class GuidedSystem(System):
        def convert_batch(self, batch, *, target, inference_cfg_rate=None):
            assert inference_cfg_rate == 0.0
            return super().convert_batch(batch, target=target)

    system = GuidedSystem()
    pipeline = SpeechConversionPipeline(
        system,
        BaseDataset(rows=[sample("src", "b")]),
        targets(value="a"),
        tmp_path,
        inference_cfg_rate=0.0,
    )
    assert pipeline.metadata["inference_cfg_rate"] == 0.0
    pipeline.run()
    assert len(system.converted) == 1
