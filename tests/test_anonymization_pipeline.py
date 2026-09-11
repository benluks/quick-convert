import pytest
import torch

from quick_convert.data import AudioSample
from quick_convert.pipelines.anonymization.pipeline import AnonymizationPipeline


class PathRecordingAnonymizer:
    sr = 16_000

    def __init__(self):
        self.inputs = []

    def anonymize(self, audio):
        self.inputs.append(audio)
        return torch.zeros(1, 8)


class SingleSampleDataset:
    root = "dataset"
    splits = None

    def __init__(self, sample):
        self.sample = sample

    def __iter__(self):
        yield self.sample


def test_anonymization_rejects_unimplemented_batching():
    with pytest.raises(NotImplementedError, match="variable-length output contract"):
        AnonymizationPipeline(
            anonymizer=object(),
            dataset=object(),
            batch_size=2,
        )


def test_anonymization_passes_a_plain_path_to_library_api(tmp_path, monkeypatch):
    from quick_convert.pipelines.anonymization import pipeline as pipeline_module

    sample = AudioSample(utt_id="sample", path=tmp_path / "input.wav")
    anonymizer = PathRecordingAnonymizer()
    monkeypatch.setattr(pipeline_module.torchaudio, "save", lambda *args, **kwargs: None)

    AnonymizationPipeline(
        anonymizer=anonymizer,
        dataset=SingleSampleDataset(sample),
        out_dir=tmp_path / "output",
    ).run()

    assert anonymizer.inputs == [sample.path]
