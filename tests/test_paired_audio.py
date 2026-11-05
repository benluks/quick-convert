from pathlib import Path

import pytest
import soundfile as sf
import torch
from torchaudio.functional import resample

from quick_convert.data import AudioBatch, load_paired_audio_batch


@pytest.mark.parametrize("max_samples", [None, 96])
def test_paired_audio_resamples_crops_and_pads_without_changing_targets(tmp_path, max_samples):
    stereo = torch.stack((torch.linspace(0, 0.8, 64), torch.linspace(0.2, 0.6, 64)), dim=1)
    mono = torch.linspace(-0.5, 0.5, 64)
    paired_paths = [tmp_path / "stereo.wav", tmp_path / "mono.wav"]
    sf.write(paired_paths[0], stereo.numpy(), 8000, subtype="FLOAT")
    sf.write(paired_paths[1], mono.numpy(), 16000, subtype="FLOAT")
    batch = AudioBatch(
        utt_ids=["a", "b"],
        paths=[tmp_path / "target-a.wav", tmp_path / "target-b.wav"],
        splits=["train", "val"],
        resources={"tts_path": [str(path) for path in paired_paths], "transcript": ["one", "two"]},
        waveforms=torch.ones(2, 20),
        lengths=torch.tensor([20, 15]),
        sample_rates=torch.tensor([24000, 24000]),
    )
    target_waveforms = batch.waveforms.clone()
    resources = dict(batch.resources)
    paths = list(batch.paths)

    paired = load_paired_audio_batch(batch, "tts_path", 16000, max_samples)

    expected = resample(stereo.mean(dim=1), 8000, 16000)[:max_samples]
    assert paired.lengths.tolist() == [expected.numel(), mono.numel()]
    assert paired.waveforms.shape == (2, expected.numel())
    torch.testing.assert_close(paired.waveforms[0], expected)
    torch.testing.assert_close(paired.waveforms[1, :64], mono)
    assert torch.count_nonzero(paired.waveforms[1, 64:]) == 0
    assert paired.waveforms.device == paired.lengths.device == paired.sample_rates.device == batch.waveforms.device
    assert paired.sample_rates.tolist() == [16000, 16000]
    assert paired.paths == paired_paths
    assert paired.utt_ids == batch.utt_ids and paired.utt_ids is not batch.utt_ids
    assert paired.splits == batch.splits and paired.splits is not batch.splits
    assert paired.resources == {} and paired.resource_refs is None
    torch.testing.assert_close(batch.waveforms, target_waveforms)
    assert batch.lengths.tolist() == [20, 15]
    assert batch.sample_rates.tolist() == [24000, 24000]
    assert batch.resources == resources
    assert batch.paths == paths


@pytest.mark.parametrize("paths", [None, [], ["a.wav"], "a.wav", Path("a.wav")])
def test_missing_or_incomplete_paired_paths_fail(paths):
    batch = AudioBatch(
        utt_ids=["a", "b"],
        paths=[Path("a.wav"), Path("b.wav")],
        splits=[None, None],
        resources={} if paths is None else {"reference_path": paths},
        waveforms=torch.ones(2, 1600),
    )
    with pytest.raises(ValueError, match="reference_path"):
        load_paired_audio_batch(batch, "reference_path", 16000)


def test_same_file_reference_is_rejected_before_loading(tmp_path, monkeypatch):
    source = tmp_path / "a.wav"
    alias = tmp_path / "alias.wav"
    alias.symlink_to(source)
    batch = AudioBatch(
        utt_ids=["a"],
        paths=[source],
        splits=[None],
        resources={"reference_path": [str(alias)]},
        waveforms=torch.ones(1, 1600),
    )

    def unexpected_load(*args, **kwargs):
        raise AssertionError("The same recording must be rejected before loading.")

    monkeypatch.setattr("quick_convert.data.paired_audio.load_audio_input", unexpected_load)
    with pytest.raises(ValueError, match="differ"):
        load_paired_audio_batch(batch, "reference_path", 16000)


def test_empty_paired_recording_is_rejected(tmp_path, monkeypatch):
    empty_path = tmp_path / "empty.wav"
    # Audio backends may reject an empty file before returning a waveform.
    # Exercise this layer's check independently of the decoder's behavior.
    monkeypatch.setattr("quick_convert.data.paired_audio.load_audio_input", lambda *a, **kw: torch.empty(1, 0))
    batch = AudioBatch(
        utt_ids=["a"],
        paths=[tmp_path / "target.wav"],
        splits=[None],
        resources={"tts_path": [str(empty_path)]},
        waveforms=torch.ones(1, 20),
    )
    with pytest.raises(ValueError, match="empty"):
        load_paired_audio_batch(batch, "tts_path", 16000)
