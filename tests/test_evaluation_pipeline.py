import pytest

from quick_convert.data import AudioBatch, AudioSample
from quick_convert.data.resources import ResourceCollection, ResourceRef
from quick_convert.pipelines.evaluation.pipeline import EvalPipeline


class BatchDataset:
    def __init__(self, batch):
        self.batch = batch

    def make_dataloader(self, **kwargs):
        return [self.batch]


class EmptyReferenceMetric:
    key = "label"

    def get_references(self, batch):
        return []


class OnePredictionSystem:
    def __init__(self):
        self.calls = 0

    def get_labels(self, batch):
        self.calls += 1
        return ["prediction"]


class FixedMetric:
    def __init__(self, key):
        self.key = key
        self.ref_key = f"ref_{key}"
        self.pred_key = f"pred_{key}"

    def get_references(self, batch):
        return [batch.utt_ids[0]]


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
    ref = AudioSample(utt_id="reference", path=tmp_path / "ref.wav")
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
            "ref_first": "reference",
            "ref_second": "reference",
            "pred_first": "prediction",
            "pred_second": "prediction",
        }
    ]
    assert system.calls == 1
