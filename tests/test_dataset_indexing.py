from pathlib import Path

import pytest
import torch
import torchaudio
from hydra.utils import instantiate
from omegaconf import OmegaConf

from quick_convert.data import BaseDataset
from quick_convert.data.recording_index import join_index, link_identities, read_rttm_intervals, segment_rows


def test_hydra_index_loads_rttm_intervals(tmp_path, monkeypatch):
    (tmp_path / "takes.csv").write_text(
        "take,reader,filename,reader_track\n"
        "take-a,alpha,a.wav,voice-1\n"
        "take-b,gamma,b.wav,voice-2\n"
        "take-c,beta,c.wav,voice-1\n"
    )
    (tmp_path / "readers.csv").write_text("reader,aliases\ngamma,beta\nbeta,alpha\nalpha,\n")
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    annotation_root = tmp_path / "annotations"
    annotation_root.mkdir()
    source = torch.arange(8000, dtype=torch.float32).unsqueeze(0) / 8000
    for filename in ("a.wav", "b.wav"):
        torchaudio.save(str(audio_root / filename), source, 8000, encoding="PCM_S", bits_per_sample=16)

    (annotation_root / "take-a.rttm").write_text(
        "SPEAKER take-a 1 0.75 0.125 <NA> <NA> voice-1 <NA> <NA>\n"
        "SPEAKER take-a 1 0.00 0.125 <NA> <NA> voice-2 <NA> <NA>\n"
        "SPEAKER take-a 1 0.25 0.125 <NA> <NA> voice-1 <NA> <NA>\n"
    )
    (annotation_root / "take-b.rttm").write_text("SPEAKER take-b 1 0.50 0.250 <NA> <NA> voice-2 <NA> <NA>\n")
    (annotation_root / "take-c.rttm").write_text("SPEAKER take-c 1 0.50 0.250 <NA> <NA> voice-2 <NA> <NA>\n")

    config = OmegaConf.load(Path(__file__).resolve().parents[1] / "docs/data/examples/rttm_dataset.yaml")
    config.data_root = str(tmp_path)

    def unexpected_decode(*args, **kwargs):
        pytest.fail("Constructing the index decoded audio.")

    with monkeypatch.context() as patches:
        patches.setattr(torchaudio, "load", unexpected_decode)
        dataset = instantiate(config.dataset)
    assert isinstance(dataset, BaseDataset)
    assert len(dataset) == 3

    batch = next(iter(dataset.make_dataloader(batch_size=3)))
    assert batch.utt_ids == ["take-a-0", "take-a-1", "take-b-0"]
    assert batch.paths == [audio_root / "a.wav", audio_root / "a.wav", audio_root / "b.wav"]
    assert batch.lengths.tolist() == [2000, 2000, 4000]
    clip = batch[0]
    assert (clip.start, clip.end, clip.sample_rate) == (0.25, 0.375, 16000)
    assert [item.resources.spkid.value for item in batch] == ["alpha"] * 3
    decoded, _ = torchaudio.load(str(audio_root / "a.wav"))
    expected = torchaudio.functional.resample(decoded[..., 2000:3000], 8000, 16000)
    torch.testing.assert_close(clip.waveform, expected.squeeze(0))


@pytest.mark.parametrize(
    ("lookup", "message"),
    [
        ([{"id": "a"}, {"id": "a"}], "Duplicate lookup"),
        ([{"id": "b"}], "No lookup row"),
        ([{"id": "a", "name": "other"}], "Conflicting values"),
    ],
)
def test_join_rejects_ambiguous_or_missing_metadata(lookup, message):
    with pytest.raises(ValueError, match=message):
        join_index([{"id": "a", "name": "original"}], lookup, key_columns=["id"])


def test_identity_link_rejects_unknown_targets():
    with pytest.raises(ValueError, match="Unknown linked identity"):
        link_identities(
            [{"id": "a", "aliases": "b"}],
            id_column="id",
            links_column="aliases",
            output_column="group",
            separator=";",
        )


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("SPEAKER take 1 0 1 <NA> <NA> other <NA> <NA>\n", "No intervals matched"),
        ("SPEAKER take 1 0 nan <NA> <NA> reader <NA> <NA>\n", "Invalid RTTM interval"),
        (
            "SPEAKER take 1 0 1 <NA> <NA> reader <NA> <NA>\nSPEAKER take 2 0 1 <NA> <NA> reader <NA> <NA>\n",
            "one recording and channel",
        ),
    ],
)
def test_rttm_index_rejects_unusable_annotations(tmp_path, contents, message):
    path = tmp_path / "take.rttm"
    path.write_text(contents)
    with pytest.raises(ValueError, match=message):
        segment_rows(
            [{"take": "take"}],
            recording_id_column="take",
            audio_path=str(tmp_path / "take.wav"),
            annotation_path=str(path),
            interval_reader=read_rttm_intervals,
            speaker="reader",
        )


def test_segment_rows_accepts_a_reader_for_another_format(tmp_path):
    audio = tmp_path / "take.wav"
    audio.touch()
    annotations = tmp_path / "take.txt"

    def read_intervals(path, speaker):
        assert path == annotations
        assert speaker == "reader"
        return [(2.0, 3.0), (0.0, 1.0)]

    rows = segment_rows(
        [{"take": "take"}],
        recording_id_column="take",
        audio_path=str(audio),
        annotation_path=str(annotations),
        interval_reader=read_intervals,
        speaker="reader",
    )
    assert [(row.start, row.end) for row in rows] == [(0.0, 1.0), (2.0, 3.0)]
