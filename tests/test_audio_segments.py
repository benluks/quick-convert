import shutil
import subprocess

import pytest
import torch
import torchaudio

from quick_convert.data import ManifestDataset
from quick_convert.utils.audio import load_audio


@pytest.fixture
def recording(tmp_path):
    sample_rate = 8000
    path = tmp_path / "recording.wav"
    waveform = torch.arange(sample_rate, dtype=torch.float32).unsqueeze(0) / sample_rate
    torchaudio.save(str(path), waveform, sample_rate, encoding="PCM_S", bits_per_sample=16)
    return path, sample_rate


def test_manifest_intervals_load_and_survive_collation(tmp_path, recording):
    path, source_sr = recording
    manifest = tmp_path / "items.csv"
    manifest.write_text(
        f"utt_id,path,start,end\nclip,{path},0.125,0.5\ntail,{path},0.75,\nwhole,{path},,\n",
        encoding="utf-8",
    )
    dataset = ManifestDataset(manifest, load=["audio"], target_sr=16000)
    source, _ = load_audio(path)

    batch = next(iter(dataset.make_dataloader(batch_size=3)))
    assert batch.utt_ids == ["clip", "tail", "whole"]
    assert batch.lengths.tolist() == [6000, 4000, 16000]
    clip = batch[0]
    assert (clip.path, clip.start, clip.end, clip.sample_rate) == (path, 0.125, 0.5, 16000)
    expected = torchaudio.functional.resample(source[..., 1000:4000], source_sr, 16000)
    torch.testing.assert_close(clip.waveform, expected.squeeze(0))
    torch.testing.assert_close(clip.load_audio().waveform, source[..., 1000:4000])

    metadata = ManifestDataset(manifest)
    assert metadata.collate_fn([metadata[0]])[0].end == 0.5


def test_custom_columns_allow_an_end_without_a_start(tmp_path, recording):
    path, _ = recording
    manifest = tmp_path / "items.csv"
    manifest.write_text(f"utt_id,path,stop\nintro,{path},0.25\n", encoding="utf-8")
    dataset = ManifestDataset(manifest, end_column="stop", load=["audio"])
    assert dataset[0].waveform.shape[-1] == 2000


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (-1, None, "nonnegative"),
        (0.5, 0.25, "greater than start"),
        (0, float("nan"), "finite"),
        (0, 0.00001, "no source frames"),
        (0.5, 1.25, "exceeds the source"),
        (1, None, "exceeds the source"),
    ],
)
def test_invalid_intervals_fail(recording, start, end, message):
    path, _ = recording
    with pytest.raises(ValueError, match=message):
        load_audio(path, start=start, end=end)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="Requires FFmpeg tools")
def test_mp4_intervals_load_without_torchaudio_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(torchaudio, "list_audio_backends", lambda: ["soundfile"])
    path = tmp_path / "recording.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=s=32x32:r=10:d=4",
            "-f",
            "lavfi",
            "-i",
            "aevalsrc=0.4*sin(2*PI*(400*t+20*t*t)):s=44100:d=4",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    full, source_sr = load_audio(path)
    start, end = 2.532, 3.032
    manifest = tmp_path / "items.csv"
    manifest.write_text(f"utt_id,path,start,end\nclip,{path},{start},{end}\n", encoding="utf-8")
    dataset = ManifestDataset(manifest, load=["audio"], target_sr=16000)

    clip = dataset[0]
    expected = full[..., round(start * source_sr) : round(end * source_sr)]
    expected = torchaudio.functional.resample(expected, source_sr, 16000)
    assert clip.sample_rate == 16000
    assert clip.waveform.shape == (1, 8000)
    torch.testing.assert_close(clip.waveform, expected, atol=1e-5, rtol=1e-4)

    with pytest.raises(ValueError, match="exceeds the source"):
        load_audio(path, start=3.5, end=4.5)
