import json
from pathlib import Path

import pytest
import torch
from torch import nn

from quick_convert.components.boundaries.trajectories import nearest_trajectories, resample_trajectory, trajectory_dtw
from quick_convert.components.ssl.base import ContentFeatures
from quick_convert.data import AudioBatch, AudioSample
from quick_convert.pipelines.trajectory_analysis import TrajectoryAnalysisPipeline, _PairedReservoir


def path(length):
    t = torch.linspace(0, 1, length)
    return torch.stack((t, torch.ones_like(t), 2 * t + 1), -1)


def meta(utt, duration=100.0, speaker="a"):
    return dict(utt_id=utt, split="test", speaker_id=speaker, duration_ms=duration)


def test_descriptor_is_duration_independent_preserves_location_and_order():
    first = resample_trajectory(path(3))
    second = resample_trajectory(path(31))
    torch.testing.assert_close(first, second)
    assert not torch.allclose(first, resample_trajectory(path(31) + 3))
    assert not torch.allclose(first, resample_trajectory(path(31).flip(0)))
    with pytest.raises(ValueError, match="nonzero"):
        resample_trajectory(torch.zeros(4, 3))


def test_retrieval_excludes_whole_utterance_and_does_not_use_duration():
    bank = torch.stack(
        [
            resample_trajectory(path(3)),
            resample_trajectory(path(30)),
            resample_trajectory(path(7) + 0.01),
            resample_trajectory(path(10) + 2),
        ]
    )
    metadata = [meta("a-1"), meta("a-1"), meta("b-1", speaker="b"), meta("c-1", speaker="c")]
    result = nearest_trajectories(bank, metadata, [0], top_k=2, batch_size=1)
    assert result[0][0][0] == 2
    assert {i for i, _ in result[0]} == {2, 3}
    for index, row in enumerate(metadata):
        row["duration_ms"] = 10**index
    assert nearest_trajectories(bank, metadata, [0], top_k=2) == result
    metadata[2]["speaker_id"] = "a"
    assert nearest_trajectories(bank, metadata, [0], different_speaker=True)[0][0][0] == 3
    assert nearest_trajectories(bank[:2], metadata[:2], [0]) == [[]]


def test_dtw_finds_uneven_timing_and_diagonal_control():
    states = torch.eye(3)
    a = states[torch.tensor([0] * 4 + [1] * 4 + [2] * 4)]
    b = states[torch.tensor([0] * 2 + [1] * 7 + [2] * 3)]
    uniform = float((1 - (a * b).sum(-1)).mean())
    aligned = trajectory_dtw(a, b, band_fraction=1.0)
    assert aligned["distance"] == 0.0
    assert aligned["non_diagonal_step_fraction"] > 0
    assert trajectory_dtw(a, b, band_fraction=0.0)["distance"] == pytest.approx(uniform)
    assert trajectory_dtw(a, a)["distance"] == 0.0


def test_paired_reservoir_is_bounded_and_reproducible():
    def sample():
        result = _PairedReservoir(8, 4, torch.Generator().manual_seed(115))
        for i in range(100):
            result.add(path(3), path(3) + 1, meta(str(i), duration=i + 1), meta(str(i), duration=i + 1))
        return result

    a, b = sample(), sample()
    assert a.seen == 100
    assert len(a.metadata["detected"]) == 8
    assert a.metadata == b.metadata
    assert a.metadata["detected"] == a.metadata["random_location"]
    torch.testing.assert_close(a.banks["detected"], b.banks["detected"])
    assert a.banks["detected"].shape == (8, 4, 3)
    assert any(int(row["utt_id"]) >= 8 for row in a.metadata["detected"])


class Dataset:
    def __init__(self):
        self.samples = [
            AudioSample(
                utt_id=f"{i // 2}-{i}",
                split="test",
                path=Path(f"{i}.wav"),
                waveform=torch.ones(40 + i * 4),
                sample_rate=100,
            )
            for i in range(6)
        ]

    def make_dataloader(self, batch_size, num_workers):
        return [
            AudioBatch.from_samples(self.samples[i : i + batch_size]) for i in range(0, len(self.samples), batch_size)
        ]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, batch):
        assert len(batch) == 1 and not self.training and not torch.is_grad_enabled()
        self.calls += 1
        frames = int(batch.lengths[0]) // 2
        t = torch.arange(frames, dtype=torch.float32)
        values = torch.stack((torch.sin(t * 0.8), torch.cos(t * 0.8), torch.ones_like(t) * 2), -1)
        return ContentFeatures(
            values[None], torch.tensor([frames]), 3, "continuous", "frame", "synthetic", "test", 6, 50.0
        )


def run(path, batch_size=1, **kwargs):
    encoder = Encoder()
    report = TrajectoryAnalysisPipeline(
        Dataset(),
        encoder,
        path,
        threshold_mode="absolute",
        threshold=0.1,
        metric="l2",
        max_segments=12,
        max_queries=6,
        points=8,
        top_k=3,
        seed=115,
        batch_size=batch_size,
        speaker_regex=r"^([^-]+)-",
        **kwargs,
    ).run()
    return report, encoder


def test_online_workflow_controls_and_batch_independence(tmp_path, monkeypatch):
    def reject_save(*args, **kwargs):
        raise AssertionError("No tensor persistence allowed")

    monkeypatch.setattr(torch, "save", reject_save)
    first, encoder = run(tmp_path / "one")
    second, _ = run(tmp_path / "two", batch_size=3)
    assert first == second
    assert encoder.calls == 6
    assert first["encoder"]["encoder_layer"] == 6
    conditions = first["populations"]["detected"]["conditions"]
    assert (
        conditions["nearest"]["cosine_distance"]["mean"]
        < conditions["time_shuffled_nearest"]["cosine_distance"]["mean"]
    )
    assert (
        first["control_duration_relationships"]["detected"]["path_length_l2"]["pooled"]["pairs"]
        == first["eligible_detected_segments"]
    )
    assert first["retained_pairs"] <= 12
    assert first["descriptor_storage_bytes"] == 2 * 12 * 8 * 3 * 4
    assert (
        first["paired_control_measurements"]["detected"]["path_length_l2"]["count"]
        == first["eligible_detected_segments"]
    )
    rows = [json.loads(line) for line in (tmp_path / "one" / "trajectory_matches.jsonl").read_text().splitlines()]
    assert rows
    for row in rows:
        assert row["query"]["utt_id"] != row["candidate"]["utt_id"]
        assert row["duration_ratio"] >= 1.0
        assert row["eligible_candidates"] > 0
    assert {row["condition"] for row in rows} == {"nearest", "dtw_refined", "random_partner", "time_shuffled_nearest"}
    assert {row["population"] for row in rows} == {"detected", "random_location"}
    # Paired banks select the same frame lengths and query identities.
    detected = [row for row in rows if row["condition"] == "nearest" and row["population"] == "detected"]
    control = [row for row in rows if row["condition"] == "nearest" and row["population"] == "random_location"]
    for a, b in zip(detected, control, strict=True):
        assert a["query"]["utt_id"] == b["query"]["utt_id"]
        assert (
            a["query"]["end_frame"] - a["query"]["start_frame"] == b["query"]["end_frame"] - b["query"]["start_frame"]
        )
        assert a["query"]["duration_ms"] == pytest.approx(b["query"]["duration_ms"])
    assert {p.suffix for p in (tmp_path / "one").iterdir()} <= {".json", ".jsonl"}
    with pytest.raises(FileExistsError):
        run(tmp_path / "one")


def test_different_speaker_filter_and_insufficient_data(tmp_path):
    _, _ = run(tmp_path / "cross", different_speaker=True)
    rows = [json.loads(line) for line in (tmp_path / "cross" / "trajectory_matches.jsonl").read_text().splitlines()]
    assert rows and all(row["same_speaker"] is False for row in rows)
    with pytest.raises(ValueError, match="Too few"):
        run(tmp_path / "few", max_utterances=1)
