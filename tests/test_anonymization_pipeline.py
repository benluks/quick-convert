import torch

from quick_convert.data import AudioBatch, AudioSample, GeneratedAudio
from quick_convert.pipelines.anonymization.pipeline import AnonymizationPipeline


class BatchAnonymizer:
    sample_rate = 24_000

    def __init__(self):
        self.batches = []

    def anonymize_batch(self, batch):
        self.batches.append(batch)
        return GeneratedAudio(
            waveforms=torch.tensor([[1.0] * 8, [2.0] * 8]),
            lengths=torch.tensor([8, 5]),
            sample_rate=self.sample_rate,
        )


class BatchDataset:
    root = "dataset"
    splits = ["test"]

    def __init__(self, batch):
        self.batch = batch
        self.loader_kwargs = None

    def make_dataloader(self, **kwargs):
        self.loader_kwargs = kwargs
        return [self.batch]


def test_anonymization_uses_batch_contract_and_trims_outputs(tmp_path, monkeypatch):
    from quick_convert.pipelines.anonymization import pipeline as pipeline_module

    samples = [
        AudioSample(utt_id="first", path=tmp_path / "first.wav", split="test"),
        AudioSample(utt_id="second", path=tmp_path / "second.wav", split="test"),
    ]
    batch = AudioBatch.from_samples(samples)
    dataset = BatchDataset(batch)
    anonymizer = BatchAnonymizer()
    saved = []
    monkeypatch.setattr(
        pipeline_module.torchaudio,
        "save",
        lambda path, waveform, sample_rate: saved.append((path, waveform.clone(), sample_rate)),
    )

    AnonymizationPipeline(
        anonymizer=anonymizer,
        dataset=dataset,
        out_dir=tmp_path / "output",
        batch_size=2,
        num_workers=3,
    ).run()

    assert dataset.loader_kwargs == {"batch_size": 2, "num_workers": 3}
    assert anonymizer.batches == [batch]
    assert [waveform.shape for _, waveform, _ in saved] == [(1, 8), (1, 5)]
    assert [sample_rate for _, _, sample_rate in saved] == [24_000, 24_000]
