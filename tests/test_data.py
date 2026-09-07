from dataclasses import replace

import torch

from quick_convert.data import AudioBatch, AudioSample, BaseDataset
from quick_convert.data.resources import ResourceCollection, ResourceRef


def test_dataset_can_be_constructed_from_paths(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.touch()

    dataset = BaseDataset(
        paths=[audio_path],
        file_format="wav",
        get_utt_id_fn=lambda path: path.stem,
    )

    assert len(dataset) == 1
    assert dataset.rows[0].utt_id == "sample"
    assert dataset.rows[0].path == audio_path


def test_dataset_preserves_speaker_ids_through_batching(tmp_path):
    audio_path = tmp_path / "speaker-1" / "sample.wav"
    audio_path.parent.mkdir()
    audio_path.touch()

    dataset = BaseDataset(
        paths=[audio_path],
        file_format="wav",
        get_utt_id_fn=lambda path: path.stem,
        get_spkid_fn=lambda path: path.parent.name,
        return_spkid=True,
    )

    batch = AudioBatch.from_samples([AudioSample(**vars(dataset.rows[0]))])

    assert dataset.rows[0].spk_id == "speaker-1"
    assert batch.spk_ids == ["speaker-1"]
    assert batch[0].spk_id == "speaker-1"


def test_audio_batch_collates_audio_and_tensor_resources():
    samples = [
        AudioSample(
            utt_id="a",
            path=None,
            waveform=torch.tensor([[1.0, 2.0, 3.0]]),
            sample_rate=16_000,
            resources=ResourceCollection.from_refs(
                [ResourceRef(name="content", kind="torch_tensor", value=torch.ones(2, 3))]
            ),
        ),
        AudioSample(
            utt_id="b",
            path=None,
            waveform=torch.tensor([[4.0, 5.0]]),
            sample_rate=16_000,
            resources=ResourceCollection.from_refs(
                [ResourceRef(name="content", kind="torch_tensor", value=torch.ones(1, 3))]
            ),
        ),
    ]

    batch = AudioBatch.from_samples(samples)

    assert batch.waveforms.shape == (2, 3)
    assert batch.lengths.tolist() == [3, 2]
    assert batch.resources["content"].values.shape == (2, 2, 3)
    assert batch.resources["content"].lengths.tolist() == [2, 1]


def test_audio_batch_from_paths_without_resources(monkeypatch, tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.touch()

    def fake_load_audio(sample, **kwargs):
        return replace(sample, waveform=torch.ones(1, 4), sample_rate=16_000)

    monkeypatch.setattr(AudioSample, "load_audio", fake_load_audio)

    batch = AudioBatch.from_paths(audio_path)

    assert batch.utt_ids == ["sample"]
    assert batch.resources == {}
    assert batch.waveforms.shape == (1, 4)
