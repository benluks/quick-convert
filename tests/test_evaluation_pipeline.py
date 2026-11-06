import json
from pathlib import Path

import pytest

from quick_convert.data import AudioBatch, AudioSample
from quick_convert.data.resources import ResourceCollection, ResourceRef
from quick_convert.pipelines.evaluation.pipeline import EvalPipeline


class BatchDataset:
    def __init__(self, *batches):
        self.batches = batches

    def make_dataloader(self, **kwargs):
        return self.batches


class EmptyReferenceMetric:
    key = "label"

    def get_references(self, batch):
        return []


class OnePredictionSystem:
    def __init__(self):
        self.calls = 0

    def get_labels(self, batch):
        self.calls += 1
        return ["prediction"] * len(batch)


class FixedMetric:
    def __init__(self, key):
        self.key = key
        self.ref_key = f"ref_{key}"
        self.pred_key = f"pred_{key}"

    def get_references(self, batch):
        return [Path(sample.path).stem for sample in batch]


def test_evaluation_records_requested_resources(tmp_path):
    sample = AudioSample(
        utt_id="sample",
        path=tmp_path / "sample.wav",
        split="test",
        resources=ResourceCollection.from_refs([ResourceRef(name="speaker", kind="text", value="speaker-1")]),
    )
    batch = AudioBatch.from_samples([sample])
    pipeline = EvalPipeline(
        dataset=BatchDataset(batch),
        system=None,
        metrics=[],
        out_dir=tmp_path,
        batch_size=1,
        record_resources=["speaker"],
    )

    assert pipeline.generate_records() == [
        {
            "utt_id": "sample",
            "path": str(sample.path),
            "split": "test",
            "speaker": "speaker-1",
        }
    ]


def test_evaluation_rejects_resource_columns_that_replace_sample_metadata(tmp_path):
    with pytest.raises(ValueError, match="conflict with sample metadata"):
        EvalPipeline(
            dataset=None,
            system=None,
            metrics=[],
            out_dir=tmp_path,
            batch_size=1,
            record_resources=["utt_id"],
        )


def test_evaluation_validates_reference_count_without_separate_reference_dataset(tmp_path):
    sample = AudioSample(utt_id="sample", path=tmp_path / "sample.wav")
    pipeline = EvalPipeline(
        dataset=BatchDataset(AudioBatch.from_samples([sample])),
        system=OnePredictionSystem(),
        metrics=[EmptyReferenceMetric()],
        out_dir=tmp_path,
        batch_size=1,
    )

    with pytest.raises(ValueError, match="Reference data returned 0 values"):
        pipeline.generate_records()


def test_evaluation_runs_inference_and_consumes_reference_batch_once(tmp_path):
    pred = AudioSample(utt_id="prediction", path=tmp_path / "pred.wav")
    ref = AudioSample(utt_id="prediction", path=tmp_path / "ref.wav")
    system = OnePredictionSystem()
    pipeline = EvalPipeline(
        dataset=BatchDataset(AudioBatch.from_samples([pred])),
        ref_dataset=BatchDataset(AudioBatch.from_samples([ref])),
        system=system,
        metrics=[FixedMetric("first"), FixedMetric("second")],
        out_dir=tmp_path,
        batch_size=1,
    )

    assert pipeline.generate_records() == [
        {
            "utt_id": "prediction",
            "path": str(pred.path),
            "split": None,
            "ref_first": "ref",
            "ref_second": "ref",
            "pred_first": "prediction",
            "pred_second": "prediction",
        }
    ]
    assert system.calls == 1


def make_batch(*ids, prefix=""):
    return AudioBatch.from_samples([AudioSample(utt_id=utt_id, path=f"{prefix}{utt_id}.wav") for utt_id in ids])


def test_evaluation_pairs_reordered_references_across_batches(tmp_path):
    pipeline = EvalPipeline(
        dataset=BatchDataset(make_batch("a", "b"), make_batch("c")),
        ref_dataset=BatchDataset(make_batch("c", "b", prefix="ref-"), make_batch("a", prefix="ref-")),
        system=OnePredictionSystem(),
        metrics=[FixedMetric("label")],
        out_dir=tmp_path,
        batch_size=2,
    )
    assert [(row["utt_id"], row["ref_label"]) for row in pipeline.generate_records()] == [
        ("a", "ref-a"),
        ("b", "ref-b"),
        ("c", "ref-c"),
    ]


@pytest.mark.parametrize(
    ("pred_ids", "ref_ids", "message"),
    [
        (("a",), ("a", "a"), "Duplicate reference"),
        (("a", "a"), ("a",), "Duplicate prediction"),
        (("a", "b"), ("a",), "Missing reference.*b"),
        (("a",), ("a", "b"), "no prediction.*b"),
    ],
)
def test_evaluation_rejects_invalid_pairing(tmp_path, pred_ids, ref_ids, message):
    pipeline = EvalPipeline(
        dataset=BatchDataset(*(make_batch(utt_id) for utt_id in pred_ids)),
        ref_dataset=BatchDataset(*(make_batch(utt_id) for utt_id in ref_ids)),
        system=OnePredictionSystem(),
        metrics=[FixedMetric("label")],
        out_dir=tmp_path,
        batch_size=1,
    )
    with pytest.raises(ValueError, match=message):
        pipeline.run()
    assert not (tmp_path / "predictions.csv").exists()


def test_ser_distance_uses_explicit_reference_system(tmp_path):
    from quick_convert.pipelines.evaluation.metrics import MeanEuclideanDistanceMetric

    class EmotionSystem:
        def get_labels(self, batch):
            return [[1.0, 0.0] if Path(sample.path).stem == "ref-b" else [0.0, 1.0] for sample in batch]

    system = EmotionSystem()
    pipeline = EvalPipeline(
        dataset=BatchDataset(make_batch("a", "b")),
        ref_dataset=BatchDataset(make_batch("b", "a", prefix="ref-")),
        system=system,
        metrics=[MeanEuclideanDistanceMetric()],
        reference_systems={"ser_embedding": system},
        out_dir=tmp_path,
        batch_size=2,
    )
    pipeline.log_config({"seed": 17, "ref_dataset": {"manifest_path": "references.csv"}})
    assert pipeline.run()["ser_med"] == pytest.approx(2**0.5 / 2)
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["schema_version"] == 1
    assert provenance["protocol"] == "utterance"
    assert provenance["pairing"] == "utt_id"
    assert provenance["resolved_config"]["ref_dataset"]["manifest_path"] == "references.csv"
    assert provenance["models"]["reference"]["ser_embedding"] == provenance["models"]["prediction"]
    assert provenance["metrics"][0]["key"] == "ser_embedding"
    with (tmp_path / "predictions.csv").open(newline="") as file:
        import csv

        rows = list(csv.DictReader(file))
    assert [float(row["ser_med"]) for row in rows] == pytest.approx([0, 2**0.5])


def test_ser_distance_rejects_broadcasting():
    from quick_convert.pipelines.evaluation.metrics import MeanEuclideanDistanceMetric

    with pytest.raises(ValueError, match="matching emotion vectors"):
        MeanEuclideanDistanceMetric().compute([1, 2], [1])


def test_unavailable_uar_fails_explicitly():
    from quick_convert.pipelines.evaluation.metrics.emo.uar import UAR

    with pytest.raises(NotImplementedError, match="ground-truth emotion labels"):
        UAR()
