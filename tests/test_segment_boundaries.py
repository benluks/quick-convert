import json
from pathlib import Path

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from quick_convert.data.recording_index import clip_segment_rows
from quick_convert.data.types import MetadataSample


def test_clipping_preserves_ids_order_inputs_and_audits_once_per_recording(tmp_path):
    source = tmp_path / "recording.mp4"
    rows = [
        MetadataSample(utt_id="take-0", path=source, start=0.0, end=1.0),
        MetadataSample(utt_id="take-1", path=source, start=97.713, end=100.793),
        MetadataSample(utt_id="take-2", path=source, start=100.736, end=101.0),
        MetadataSample(utt_id="take-3", path=source, start=102.0, end=103.0),
        MetadataSample(utt_id="take-4", path=source, start=5.0, end=6.0),
    ]
    probes = []

    def duration(path):
        probes.append(path)
        return 100.736

    report_path = tmp_path / "audit" / "boundaries.json"
    result = clip_segment_rows(rows, duration_reader=duration, report_path=report_path)
    assert probes == [source]
    assert [row.utt_id for row in result] == ["take-0", "take-1", "take-4"]
    assert result[1].end == 100.736
    assert result[1].resources is rows[1].resources
    assert rows[1].end == 100.793
    assert result[0] is rows[0]
    report = json.loads(report_path.read_text())
    assert report["input_segments"] == 5
    assert report["output_segments"] == 3
    assert report["clipped_segments"] == 1
    assert report["excluded_segments"] == 2
    assert report["changes"][0]["removed_seconds"] == pytest.approx(0.057)
    assert [change["action"] for change in report["changes"]] == ["clip", "exclude", "exclude"]


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_bad_duration_fails_without_overwriting_report(tmp_path, duration):
    report = tmp_path / "report.json"
    report.write_text("previous audit")
    row = MetadataSample(utt_id="a", path=Path("a.wav"), start=0.0, end=1.0)
    with pytest.raises(ValueError, match="Invalid audio duration"):
        clip_segment_rows([row], duration_reader=lambda path: duration, report_path=report)
    assert report.read_text() == "previous audit"


def test_all_excluded_still_writes_audit(tmp_path):
    row = MetadataSample(utt_id="a", path=Path("a.wav"), start=2.0, end=3.0)
    report = tmp_path / "report.json"
    with pytest.raises(ValueError, match="No intervals remain"):
        clip_segment_rows([row], duration_reader=lambda path: 1, report_path=report)
    assert json.loads(report.read_text())["excluded_segments"] == 1


def test_hydra_wrapper_uses_audio_duration_without_decoding(tmp_path, monkeypatch):
    from quick_convert.utils import audio

    monkeypatch.setattr(audio, "probe_audio_duration", lambda path: 1.0)
    (tmp_path / "a.wav").touch()
    (tmp_path / "a.rttm").write_text("SPEAKER a 1 0.5 0.7 <NA> <NA> reader <NA> <NA>\n")
    config = OmegaConf.create(
        {
            "rows": {
                "_target_": "quick_convert.data.recording_index.clip_segment_rows",
                "rows": {
                    "_target_": "quick_convert.data.recording_index.segment_rows",
                    "rows": [{"take": "a"}],
                    "recording_id_column": "take",
                    "audio_path": str(tmp_path / "{take}.wav"),
                    "annotation_path": str(tmp_path / "{take}.rttm"),
                    "speaker": "reader",
                    "interval_reader": {
                        "_target_": "quick_convert.data.recording_index.read_rttm_intervals",
                        "_partial_": True,
                    },
                },
            },
        }
    )
    result = instantiate(config.rows)
    assert result[0].utt_id == "a-0"
    assert result[0].end == 1.0
