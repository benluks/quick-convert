from dataclasses import replace

import pytest
import soundfile as sf
import torch
from torchaudio.functional import resample

from quick_convert.data import AudioBatch, AudioSample, ManifestDataset
from quick_convert.data.resources import (
    PathResourceProvider,
    ResourceCollection,
    ResourceRef,
    collate_resources,
    load_resource,
)


@pytest.fixture
def audio_files(tmp_path):
    stereo = torch.stack((torch.linspace(0, 0.8, 64), torch.linspace(0.2, 0.6, 64)), dim=1)
    mono = torch.linspace(-0.5, 0.5, 64)
    paths = [tmp_path / "stereo.wav", tmp_path / "mono.wav"]
    sf.write(paths[0], stereo.numpy(), 8000, subtype="FLOAT")
    sf.write(paths[1], mono.numpy(), 16000, subtype="FLOAT")
    return paths, stereo, mono


def test_audio_resource_uses_standard_loading_and_collation(audio_files):
    paths, stereo, mono = audio_files
    refs = [ResourceRef("noise", "audio", path=path, sample_rate=16000) for path in paths]
    samples = [
        AudioSample(
            utt_id=str(index),
            path=path,
            split="train",
            waveform=torch.ones(1, 20),
            sample_rate=24000,
            resources=ResourceCollection.from_refs([load_resource(ref)]),
        )
        for index, (path, ref) in enumerate(zip(paths, refs, strict=True))
    ]
    batch = AudioBatch.from_samples(samples)
    noise = batch.resources["noise"]
    assert isinstance(noise, AudioBatch)
    assert noise.lengths.tolist() == [128, 64]
    assert noise.sample_rates.tolist() == [16000, 16000]
    assert noise.paths == paths
    assert noise.utt_ids == batch.utt_ids and noise.splits == batch.splits
    torch.testing.assert_close(noise.waveforms[0], resample(stereo.mean(dim=1), 8000, 16000))
    torch.testing.assert_close(noise.waveforms[1, :64], mono)
    assert noise.waveforms[1, 64:].count_nonzero() == 0
    assert noise.resources == {}
    assert batch.lengths.tolist() == [20, 20]
    assert batch.sample_rates.tolist() == [24000, 24000]
    assert torch.all(batch.waveforms == 1)
    assert all(ref.value is None for ref in refs)
    # Audio resources can share target paths; pairing restrictions are consumer policy.
    restored = batch[1].resources["noise"].value
    torch.testing.assert_close(restored.waveform, mono)
    assert restored.sample_rate == 16000
    assert restored.waveform.untyped_storage().data_ptr() == noise.waveforms.untyped_storage().data_ptr()


def test_manifest_audio_is_lazy_until_selected_and_keeps_native_rate(audio_files, tmp_path):
    paths, _, _ = audio_files
    manifest = tmp_path / "data.csv"
    manifest.write_text(f"utt_id,path,noise\na,{paths[0]},{paths[1]}\n")
    spec = {"noise": {"column": "noise", "kind": "audio"}}
    lazy = ManifestDataset(manifest, resources=spec, load=[])[0]
    assert lazy.resources["noise"].path == paths[1]
    assert lazy.resources["noise"].value is None
    dataset = ManifestDataset(manifest, resources=spec, load=["noise"])
    batch = next(iter(dataset.make_dataloader(batch_size=1)))
    assert batch.waveforms is None
    assert batch.resources["noise"].sample_rates.tolist() == [16000]
    # The same manifest supports a resource-specific target rate.
    spec["noise"]["sample_rate"] = 8000
    converted = ManifestDataset(manifest, resources=spec, load="all")[0]
    assert converted.resources["noise"].value.sample_rate == 8000
    assert converted.resources["noise"].value.waveform.shape == (1, 32)


def test_path_provider_and_from_paths_load_audio_sidecars(audio_files):
    paths, _, _ = audio_files
    provider = PathResourceProvider("noise", str(paths[0]), kind="audio", sample_rate=16000)
    batch = AudioBatch.from_paths(paths[1], resource_providers=[provider])
    assert batch.resources["noise"].lengths.tolist() == [128]
    assert batch[0].resources["noise"].value.sample_rate == 16000


def test_unloaded_or_mixed_audio_resources_fail_clearly(audio_files):
    paths, _, _ = audio_files
    ref = ResourceRef("noise", "audio", path=paths[0])
    sample = AudioSample("a", paths[0], resources=ResourceCollection.from_refs([ref]))
    with pytest.raises(ValueError, match="must be loaded"):
        collate_resources([sample])
    loaded = replace(sample, resources=ResourceCollection.from_refs([load_resource(ref)]))
    other = replace(sample, resources=ResourceCollection.from_refs([ResourceRef("noise", "text", value="other")]))
    with pytest.raises(ValueError, match="mixed resource kinds"):
        collate_resources([loaded, other])


@pytest.mark.parametrize("kind,rate", [("audio", 0), ("audio", -1), ("text", 16000)])
def test_audio_rate_options_are_validated(kind, rate):
    with pytest.raises(ValueError, match="sample_rate"):
        ResourceRef("noise", kind, value="unused", sample_rate=rate)


def test_empty_audio_resource_is_rejected(audio_files, monkeypatch):
    paths, _, _ = audio_files
    monkeypatch.setattr("quick_convert.data.resources.audio.load_audio", lambda *a: (torch.empty(1, 0), 16000))
    with pytest.raises(ValueError, match="empty"):
        load_resource(ResourceRef("noise", "audio", path=paths[0]))
