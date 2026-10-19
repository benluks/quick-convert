import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import nn

from quick_convert.components.boundaries import detect_boundaries
from quick_convert.components.boundaries.measurements import measure_segments
from quick_convert.components.ssl.base import ContentFeatures
from quick_convert.data import AudioBatch, AudioSample
from quick_convert.pipelines.boundary_analysis import BoundaryAnalysisPipeline


class Dataset:
    def __init__(self):
        self.samples = [
            AudioSample(
                utt_id=str(index),
                path=Path(f"{index}.wav"),
                split="test",
                waveform=torch.arange(size, dtype=torch.float32),
                sample_rate=100,
            )
            for index, size in enumerate((17, 23, 9))
        ]

    def make_dataloader(self, batch_size, num_workers):
        return [
            AudioBatch.from_samples(self.samples[start : start + batch_size])
            for start in range(0, len(self.samples), batch_size)
        ]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, batch):
        assert not self.training and not torch.is_grad_enabled()
        assert len(batch) == 1
        assert batch.waveforms.shape[1] == batch.lengths[0]
        self.calls += 1
        frames = int(batch.lengths[0]) * 50 // int(batch.sample_rates[0])
        # Two distinct trajectories, both derived solely from the sample.
        first = (torch.arange(frames) // 3).float()
        second = (torch.arange(frames) % 2).float()
        values = torch.stack(
            [torch.stack([first, torch.ones(frames)], -1), torch.stack([second, torch.ones(frames)], -1)], 1
        )
        return ContentFeatures(
            values[None], torch.tensor([frames]), 2, "continuous", "frame", "synthetic", "test-encoder", None, 50
        )


def run(path, batch_size=1, **kwargs):
    encoder = Encoder()
    result = BoundaryAnalysisPipeline(
        Dataset(),
        encoder,
        path,
        metrics=["l2"],
        thresholds=[0.5],
        threshold_mode="absolute",
        min_distance_ms=[20],
        batch_size=batch_size,
        **kwargs,
    ).run()
    return result, encoder


def test_online_results_match_offline_and_are_batch_independent(tmp_path, monkeypatch):
    def reject_save(*args, **kwargs):
        raise AssertionError("Embedding persistence must not occur")

    monkeypatch.setattr(torch, "save", reject_save)
    first, encoder = run(tmp_path / "one")
    second, _ = run(tmp_path / "two", batch_size=3)
    assert first == second
    assert encoder.calls == 3
    rows = [json.loads(line) for line in (tmp_path / "one" / "segments.jsonl").read_text().splitlines()]
    for group in first["groups"]:
        selected = [
            row
            for row in rows
            if row["layer_index"] == group["layer_index"] and (group["population"] == "all" or not row["is_edge"])
        ]
        durations = torch.tensor([row["duration_ms"] for row in selected], dtype=torch.float64)
        assert group["duration_ms_count"] == len(selected)
        if len(selected):
            assert group["duration_ms_mean"] == pytest.approx(float(durations.mean()))
            assert group["duration_ms_std"] == pytest.approx(float(durations.std(correction=0)))
            assert group["duration_ms_p50"] == pytest.approx(
                float(durations.median()) if len(durations) % 2 else float(durations.quantile(0.5))
            )
    for index, seconds in enumerate((0.17, 0.23, 0.09)):
        selected = [row for row in rows if row["utt_id"] == str(index) and row["layer_index"] == 0]
        assert sum(row["duration_ms"] for row in selected) == pytest.approx(seconds * 1000)
        assert selected[-1]["end_seconds"] == seconds
    assert {p.suffix for p in (tmp_path / "one").iterdir()} <= {".json", ".jsonl", ".csv"}


def test_sweep_isolates_settings_and_bounds_quantile_storage(tmp_path):
    result = BoundaryAnalysisPipeline(
        Dataset(),
        Encoder(),
        tmp_path,
        metrics=["l2", "cosine"],
        thresholds=[0.5, 0.9],
        smoothing_frames=[1, 3],
        min_distance_ms=[20, 45],
        layers=[-1],
        quantile_sample_size=2,
        write_segments=False,
    ).run()
    assert len(result["groups"]) == 2 * 2 * 2 * 2 * 2
    assert all(group["layer_index"] == 1 for group in result["groups"])
    assert all(group["duration_quantile_sample_count"] <= 2 for group in result["groups"])
    assert {g["min_distance_frames"] for g in result["groups"]} == {1, 3}
    assert not (tmp_path / "segments.jsonl").exists()
    rows = [json.loads(line) for line in (tmp_path / "utterances.jsonl").read_text().splitlines()]
    assert len(rows) == 3 * 2 * 2 * 2 * 2
    assert all(row["threshold_mode"] == "utterance_quantile" for row in rows)


def test_endpoint_timing_excludes_boundary_jumps():
    values = torch.tensor([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [3.0, 0.0], [3.0, 0.0]])
    rows = measure_segments(
        values,
        torch.tensor([0.0, 3.0, 0.0, 0.0]),
        threshold=1.0,
        min_distance_frames=1,
        frame_hz=50.0,
        duration_seconds=0.113,
    )
    assert [row["duration_ms"] for row in rows] == pytest.approx([40.0, 73.0])
    assert all(row["path_length_l2"] == 0 for row in rows)
    assert rows[0]["right_strength"] == 3
    assert rows[1]["left_strength"] == 3
    with pytest.raises(ValueError, match="past the audio"):
        measure_segments(
            values,
            torch.tensor([0.0, 0.0, 0.0, 3.0]),
            threshold=1.0,
            min_distance_frames=1,
            frame_hz=50.0,
            duration_seconds=0.06,
        )


def test_flat_peaks_and_constant_signals():
    assert detect_boundaries(torch.tensor([0.0, 3.0, 3.0, 0.0]), threshold=1.0).tolist() == [2]
    assert detect_boundaries(torch.tensor([3.0, 3.0, 0.0]), threshold=1.0).tolist() == [1]
    assert detect_boundaries(torch.tensor([0.0, 3.0, 3.0]), threshold=1.0).tolist() == [2]
    assert detect_boundaries(torch.ones(10), threshold=0.0).numel() == 0
    assert detect_boundaries(torch.ones(1), threshold=0.0).numel() == 0


def test_output_refusal_and_short_utterances(tmp_path):
    dataset = Dataset()
    dataset.samples = dataset.samples[:1]
    dataset.samples[0] = AudioSample(
        utt_id="short", path=Path("short.wav"), split="test", waveform=torch.ones(2), sample_rate=100
    )
    result = BoundaryAnalysisPipeline(dataset, Encoder(), tmp_path).run()
    assert result["skipped_short_utterances"] == 1
    assert result["groups"] == []
    with pytest.raises(FileExistsError):
        run(tmp_path)


def test_missing_frame_rate_is_rejected(tmp_path):
    class MissingRate(Encoder):
        def forward(self, batch):
            content = super().forward(batch)
            content.frame_hz = None
            return content

    with pytest.raises(ValueError, match="frame_hz"):
        BoundaryAnalysisPipeline(Dataset(), MissingRate(), tmp_path).run()


def test_run_config_resolves_dataset_overrides():
    config_dir = Path(__file__).parents[1] / "quick_convert" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(
            config_name="run/analyze_boundaries_librispeech",
            overrides=["dataset=clac", "dataset.root=/tmp/clac", "dataset.splits=[max_phonation]"],
        )
    assert config.dataset.root == "/tmp/clac"
    assert config.dataset.load is True
    assert config.dataset.target_sr == 16000
    assert config.encoder.layer is None


def test_real_dataset_and_hydra_execution(tmp_path, monkeypatch):
    import soundfile as sf
    from hydra.utils import instantiate

    from quick_convert.cli.run import execute_pipeline
    from quick_convert.pipelines import boundary_analysis

    audio_root = tmp_path / "audio"
    split = audio_root / "dev-clean"
    split.mkdir(parents=True)
    sf.write(split / "sample.flac", torch.linspace(-0.5, 0.5, 1600).numpy(), 16000)
    output = tmp_path / "output"
    config_dir = Path(__file__).parents[1] / "quick_convert" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(
            config_name="run/analyze_boundaries_librispeech",
            overrides=[
                f"dataset.root={audio_root}",
                f"pipeline.out_dir={output}",
                "encoder._target_=torch.nn.Identity",
                "+pipeline.layers=[0]",
                "+pipeline.max_utterances=1",
            ],
        )
    # Keep the actual dataset/audio loading and Hydra runner; substitute only
    # the expensive external encoder backend with a deterministic frame model.
    original = boundary_analysis.BoundaryAnalysisPipeline

    def pipeline_factory(**kwargs):
        assert isinstance(kwargs.pop("encoder"), nn.Identity)
        return original(encoder=Encoder(), **kwargs)

    monkeypatch.setattr(boundary_analysis, "BoundaryAnalysisPipeline", pipeline_factory)
    dataset = instantiate(config.dataset)
    assert dataset[0].sample_rate == 16000
    result = execute_pipeline(config)
    assert result["processed_utterances"] == 1
    assert (output / "config.yaml").is_file()
    assert (output / "summary.csv").is_file()
