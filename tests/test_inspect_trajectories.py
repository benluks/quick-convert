import json
from xml.etree import ElementTree

import pytest
import soundfile as sf
import torch
from torch import nn

from quick_convert.components.boundaries.inspection import compare_trajectories, select_pairs
from quick_convert.components.boundaries.trajectories import resample_trajectory
from quick_convert.components.ssl.base import ContentFeatures
from quick_convert.data import AudioBatch
from quick_convert.pipelines.inspect_trajectories import InspectTrajectoriesPipeline, _heatmap


def segment(utt, start=2, end=8):
    return dict(
        utt_id=utt,
        split="test",
        path=f"/original/{utt}.wav",
        start_frame=start,
        end_frame=end,
        start_seconds=start / 50,
        end_seconds=end / 50,
        duration_ms=(end - start) / 50 * 1000,
        speaker_id=utt,
    )


def record(i, ratio=2.0):
    return dict(
        population="detected",
        condition="nearest",
        query=segment(f"q{i}"),
        candidate=segment(f"c{i}"),
        cosine_distance=0.1 + i * 0.05,
        duration_ratio=ratio,
    )


def test_selection_covers_scores_and_duration_buckets_without_duplicates():
    rows = [record(i, ratio=1.2 if i % 2 == 0 else 2.0) for i in range(12)]
    selected = select_pairs(rows + [rows[0]], score_bins=3, pairs_per_stratum=1)
    assert len(selected) == 6
    assert {p["score_stratum"] for p in selected} == {1, 2, 3}
    assert {p["duration_bucket"] for p in selected} == {"smaller_difference", "larger_difference"}
    assert len({p["record"]["query"]["utt_id"] for p in selected}) == 6
    assert select_pairs(rows) == select_pairs(list(reversed(rows)))
    with pytest.raises(ValueError, match="No saved matches"):
        select_pairs(rows, population="random_location")


def test_diagnostics_score_and_known_stretch_reference():
    t = torch.linspace(0, 1, 11)
    q = torch.stack((t, torch.ones_like(t), 2 * t + 1), -1)
    c = q + 1.0
    result = compare_trajectories(q, c, points=16, max_display_frames=5)
    expected = float((1 - (resample_trajectory(q) * resample_trajectory(c)).sum(-1)).mean())
    assert result["cosine_distance"] == pytest.approx(expected, abs=1e-6)
    assert len(result["native_distance_matrix"]) == 5
    assert len(result["native_distance_matrix"][0]) == 5
    assert len(result["pointwise_distances"]) == 16
    assert all(abs(ref["cosine_distance"]) < 1e-6 for ref in result["stretch_references"])
    assert result["cosine_distance"] > 0.001
    ElementTree.fromstring(_heatmap(result["normalized_distance_matrix"], "<unsafe & title>"))
    with pytest.raises(ValueError):
        compare_trajectories(q, c, stretch_factors=[1.0])


class Encoder(nn.Module):
    sample_rate = 100

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, batch):
        assert not self.training and not torch.is_grad_enabled()
        self.calls += 1
        frames = int(batch.lengths[0]) // 2
        t = torch.linspace(0, 1, frames)
        # Full-utterance duration/context affects selected frames. Encoding a
        # cropped waveform instead of the source cannot reproduce the score.
        offset = float(batch.waveforms.mean())
        values = torch.stack((t + offset, torch.ones_like(t), t.square() + offset), -1)
        return ContentFeatures(
            values[None], torch.tensor([frames]), 3, "continuous", "frame", "synthetic", "inspection-test", 6, 50.0
        )


def fixture_files(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    encoder = Encoder().eval()
    features = {}
    for i, utt in enumerate(["q", "c", "r"]):
        sf.write(audio / f"{utt}.wav", torch.full((400,), 0.1 + i * 0.1).numpy(), 100)
        with torch.inference_mode():
            features[utt] = encoder(AudioBatch.from_paths(audio / f"{utt}.wav", target_sr=100, mono=True)).values[0]
    q, c, r = segment("q"), segment("c", 10, 19), segment("r", 20, 25)
    rows = []
    for condition, other in [("nearest", c), ("random_partner", r)]:
        score = compare_trajectories(
            features["q"][2:8], features[other["utt_id"]][other["start_frame"] : other["end_frame"]]
        )["cosine_distance"]
        rows.append(
            dict(
                population="detected",
                condition=condition,
                query=q,
                candidate=other,
                cosine_distance=score,
                duration_ratio=max(q["duration_ms"], other["duration_ms"])
                / min(q["duration_ms"], other["duration_ms"]),
            )
        )
    source = tmp_path / "run"
    source.mkdir()
    (source / "trajectory_matches.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (source / "trajectory_analysis.json").write_text(
        json.dumps(
            dict(
                descriptor_points=16,
                encoder=dict(model_name="inspection-test", backend="synthetic", encoder_layer=6, frame_hz=50.0),
            )
        )
    )
    encoder.calls = 0
    return source, audio, encoder


def test_inspection_reencodes_only_unique_sources_and_writes_portable_report(tmp_path, monkeypatch):
    source, audio, encoder = fixture_files(tmp_path)

    def reject_save(*args, **kwargs):
        raise AssertionError("No raw tensor persistence")

    monkeypatch.setattr(torch, "save", reject_save)
    output = tmp_path / "report"
    pipeline = InspectTrajectoriesPipeline(source, encoder, output, path_prefix_from="/original", path_prefix_to=audio)
    report = pipeline.run()
    assert encoder.calls == 3
    assert report["encoded_utterances"] == 3
    assert len(report["pairs"]) == 1
    pair = report["pairs"][0]
    assert pair["saved_score_absolute_difference"] < 1e-6
    assert pair["random_diagnostics"] is not None
    assert report["descriptor_points"] == 16
    page = (output / "inspection.html").read_text()
    assert "data:audio/wav;base64," in page
    assert 'src="http' not in page and "<script" not in page
    assert "Normalized real pair" in page and "Known self stretch" in page
    assert (output / "inspection.json").is_file()
    assert not list(output.glob("*.pt"))
    with pytest.raises(FileExistsError):
        pipeline.run()


def test_reproducibility_failure_does_not_write_report(tmp_path):
    source, audio, encoder = fixture_files(tmp_path)
    path = source / "trajectory_matches.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["cosine_distance"] += 0.1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="Recomputed score differs"):
        InspectTrajectoriesPipeline(source, encoder, output, path_prefix_from="/original", path_prefix_to=audio).run()
    assert not (output / "inspection.html").exists()
    assert not (output / "inspection.json").exists()


def test_encoder_mismatch_is_rejected(tmp_path):
    source, audio, encoder = fixture_files(tmp_path)
    path = source / "trajectory_analysis.json"
    report = json.loads(path.read_text())
    report["encoder"]["encoder_layer"] = 7
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="metadata differs"):
        InspectTrajectoriesPipeline(
            source,
            encoder,
            tmp_path / "report",
            path_prefix_from="/original",
            path_prefix_to=audio,
            include_audio=False,
        ).run()
