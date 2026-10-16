from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from quick_convert.components.boundaries.duration import composition_moments, duration_labels, fit_frame_bins
from quick_convert.components.ssl.base import ContentFeatures
from quick_convert.data import AudioBatch, AudioSample
from quick_convert.pipelines.duration_probe import (
    DurationProbePipeline,
    ExtractDurationCompositionPipeline,
    split_speakers,
)
from quick_convert.training.duration_probe import fit_duration_probe, predict_probabilities, probability_metrics


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_moments_use_original_frames_without_interpolation():
    values = torch.tensor([[0.0, 1.0], [0.0, 3.0], [9.0, 2.0]])
    moments = composition_moments(values)
    torch.testing.assert_close(moments[0], values.mean(0))
    torch.testing.assert_close(moments[1], values.std(0, correction=0))
    torch.testing.assert_close(moments, composition_moments(values.repeat_interleave(3, dim=0)))
    with pytest.raises(ValueError):
        composition_moments(values[:1])


def test_integer_bins_train_only_inclusive_edges_and_tail():
    train = torch.tensor([2, 2, 3, 3, 4, 4, 5, 6, 8, 10, 12, 20])
    edges = fit_frame_bins(train, n_bins=4)
    labels = duration_labels(torch.cat((edges, torch.tensor([1000]))), edges)
    assert labels[:-1].tolist() == list(range(len(edges)))
    assert labels[-1] == len(edges)
    assert edges.max() < train.max()
    assert fit_frame_bins(torch.full((10,), 2), 12).numel() == 0
    with pytest.raises(ValueError):
        fit_frame_bins(torch.tensor([1.0, 2.0]))


def test_speaker_splits_are_disjoint_and_order_independent():
    ids = [f"s{i}" for i in range(10)]
    a = split_speakers(ids)
    b = split_speakers(list(reversed(ids)))
    assert a == b
    assert [len(a[key]) for key in ("train", "validation", "test")] == [6, 2, 2]
    assert not set(a["train"]) & set(a["test"])
    with pytest.raises(ValueError):
        split_speakers(["a", "b"])


def test_probe_learns_known_conditional_categories_and_train_normalization():
    x = torch.tensor([-2.0, 0.0, 2.0]).repeat(40)[:, None]
    y = torch.tensor([0, 1, 2]).repeat(40)
    model, center, scale, history = fit_duration_probe(
        dict(train=x, validation=x[:30], test=torch.full((1, 1), 999.0)),
        dict(train=y, validation=y[:30]),
        torch.ones(3) / 3,
        hidden_dim=16,
        dropout=0.0,
        max_steps=200,
        batch_size=64,
        learning_rate=0.03,
        weight_decay=0.0,
        eval_interval=20,
        patience=10,
    )
    assert center == pytest.approx(x.mean(0))
    torch.testing.assert_close(scale, x.std(0, correction=0))
    probabilities = predict_probabilities(model, (x[:30] - center) / scale, "cpu")
    metrics = probability_metrics(probabilities, y[:30], ["heldout"] * 30)
    assert metrics["nll"] < 0.1 and metrics["accuracy"] > 0.95
    assert history["best_step"] > 0
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(30))


class Dataset:
    def __init__(self):
        self.samples = [
            AudioSample(
                utt_id=f"{speaker}-{utterance}",
                path=Path(f"{speaker}-{utterance}.wav"),
                split="source",
                waveform=torch.ones(80 + utterance * 8),
                sample_rate=100,
            )
            for speaker in range(6)
            for utterance in range(2)
        ]
        self.rows = [SimpleNamespace(utt_id=s.utt_id, path=s.path, split=s.split) for s in self.samples]

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
        n = int(batch.lengths[0]) // 2
        t = torch.arange(n, dtype=torch.float32)
        # Uneven transitions provide both short and long detector segments.
        state = torch.sin(t * 0.55) + torch.sin(t * 0.2)
        values = torch.stack((state, torch.cos(t * 0.55), torch.ones(n) * 2), -1)
        return ContentFeatures(values[None], torch.tensor([n]), 3, "continuous", "frame", "test", "synthetic", 6, 50.0)


def extract(path, batch_size=1):
    encoder = Encoder()
    extractor = ExtractDurationCompositionPipeline(
        Dataset(),
        encoder,
        path,
        projection_dim=2,
        train_capacity=20,
        heldout_capacity=10,
        threshold_mode="absolute",
        threshold=0.3,
        metric="l2",
        aggregation_sizes=(1, 2, 3),
        audit_capacity=5,
        batch_size=batch_size,
    )
    extractor.run()
    return torch.load(path, weights_only=True), encoder


def test_online_cache_is_bounded_speaker_disjoint_and_batch_independent(tmp_path):
    first, encoder = extract(tmp_path / "one.pt")
    second, _ = extract(tmp_path / "two.pt", batch_size=4)
    assert encoder.calls == 12 and first["processed_utterances"] == 12
    assert first["speaker_splits"] == second["speaker_splits"]
    for size, partitions in first["groups"].items():
        utterances = {}
        for partition, data in partitions.items():
            assert data["features"].shape[1:] == (2, 2)
            assert len(data["features"]) <= (20 if partition == "train" else 10)
            assert set(data["speakers"]) <= set(first["speaker_splits"][partition])
            torch.testing.assert_close(data["features"], second["groups"][size][partition]["features"])
            torch.testing.assert_close(data["frames"], second["groups"][size][partition]["frames"])
            assert data["frames"].dtype == torch.long
            utterances[partition] = set(data["utterances"])
        assert not utterances["train"] & utterances["test"]
    assert first["audit"]["features"].shape == (5, 4, 2, 2)
    with pytest.raises(FileExistsError):
        extract(tmp_path / "one.pt")


def test_cached_training_reports_all_aggregations_and_controls(tmp_path):
    cache, _ = extract(tmp_path / "cache.pt")
    result = DurationProbePipeline(
        tmp_path / "fit",
        cache_path=tmp_path / "cache.pt",
        max_steps=10,
        eval_interval=5,
        patience=2,
        n_bins=4,
        hidden_dim=8,
        train_batch_size=8,
    ).run()
    assert set(result["aggregation_results"]) == {"1", "2", "3"}
    for size, state in result["aggregation_results"].items():
        expected = fit_frame_bins(cache["groups"][size]["train"]["frames"], 4).tolist()
        assert state["bin_upper_edges_frames"] == expected
        assert set(state["models"]) == {"mean", "variation", "both", "both_shuffled_within_speaker"}
        for name, model in state["models"].items():
            scores = model["metrics"]["test"]
            assert scores["nll"] >= 0 and scores["brier"] >= 0
            checkpoint = torch.load(tmp_path / "fit" / "models" / f"aggregate{size}_{name}.pt", weights_only=True)
            assert checkpoint["normalization_mean"].numel() in (2, 4)
            assert checkpoint["bin_upper_edges_frames"].tolist() == expected
            assert model["resampling_audit"] is not None if size == "1" else model["resampling_audit"] is None
    assert (tmp_path / "fit" / "duration_report.json").is_file()
    with pytest.raises(FileExistsError):
        DurationProbePipeline(tmp_path / "fit", cache_path=tmp_path / "cache.pt").run()


def test_cache_partition_leakage_rejected_before_training(tmp_path):
    cache, _ = extract(tmp_path / "cache.pt")
    cache["speaker_splits"]["test"].append(cache["speaker_splits"]["train"][0])
    torch.save(cache, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="Speaker leakage"):
        DurationProbePipeline(tmp_path / "fit", cache_path=tmp_path / "bad.pt").run()
    assert not (tmp_path / "fit" / "models").exists()


def test_online_orchestrator_and_cached_run_agree_without_encoder_rerun(tmp_path):
    encoder = Encoder()
    extractor = ExtractDurationCompositionPipeline(
        Dataset(),
        encoder,
        tmp_path / "descriptors.pt",
        projection_dim=2,
        train_capacity=20,
        heldout_capacity=10,
        threshold_mode="absolute",
        threshold=0.3,
        metric="l2",
        aggregation_sizes=(1,),
        audit_capacity=3,
    )
    kwargs = dict(max_steps=5, eval_interval=5, hidden_dim=8, n_bins=3, train_batch_size=8)
    first = DurationProbePipeline(tmp_path / "online", extractor=extractor, **kwargs).run()
    assert encoder.calls == 12
    second = DurationProbePipeline(tmp_path / "cached", cache_path=tmp_path / "descriptors.pt", **kwargs).run()
    assert encoder.calls == 12
    assert first["aggregation_results"] == second["aggregation_results"]


def test_no_test_features_affect_fitting_and_zero_step_is_frequency_baseline():
    x = torch.tensor([[-1.0], [0.0], [1.0]]).repeat(6, 1)
    labels = torch.tensor([0, 1, 2]).repeat(6)
    kwargs = dict(hidden_dim=4, dropout=0.0, max_steps=2, eval_interval=1, patience=2, learning_rate=0.01, seed=115)
    a, mean_a, scale_a, _ = fit_duration_probe(
        dict(train=x, validation=x, test=x), dict(train=labels, validation=labels), torch.ones(3) / 3, **kwargs
    )
    b, mean_b, scale_b, _ = fit_duration_probe(
        dict(train=x, validation=x, test=x * 999), dict(train=labels, validation=labels), torch.ones(3) / 3, **kwargs
    )
    torch.testing.assert_close(mean_a, mean_b)
    torch.testing.assert_close(scale_a, scale_b)
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name])
