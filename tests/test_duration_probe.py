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


def extract(path, batch_size=1, **kwargs):
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
        **kwargs,
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


def test_duration_errors_distinguish_intervals_point_estimates_and_order():
    from quick_convert.training.duration_probe import duration_error_metrics, fit_frame_distribution

    edges = torch.tensor([4, 8])
    distribution = fit_frame_distribution(torch.tensor([2, 4, 6, 8, 10, 20]), edges)
    # Correct bins have zero interval error; empirical medians need not equal targets.
    probs = torch.eye(3)
    scores = duration_error_metrics(probs, torch.tensor([4, 8, 100]), edges, distribution, 50)
    assert scores["modal_bin_interval_mae_frames"] == 0
    assert scores["mixture_median_mae_frames"] == pytest.approx(94 / 3)
    assert scores["mixture_median_mae_ms"] == pytest.approx(94 / 3 * 20)
    assert scores["ranked_probability_score"] == 0
    near = duration_error_metrics(torch.tensor([[0.0, 1.0, 0.0]]), torch.tensor([2]), edges, distribution, 50)
    far = duration_error_metrics(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([2]), edges, distribution, 50)
    assert near["modal_bin_interval_mae_frames"] == 3
    assert far["modal_bin_interval_mae_frames"] == 7
    assert near["ranked_probability_score"] < far["ranked_probability_score"]
    # Median is taken across the full probability mixture, not the modal bin.
    mixed = duration_error_metrics(torch.tensor([[0.4, 0.35, 0.25]]), torch.tensor([6]), edges, distribution, 50)
    assert mixed["mixture_median_mae_frames"] == 0


def test_saved_duration_evaluation_agrees_and_supports_legacy_checkpoint(tmp_path):
    from quick_convert.pipelines.evaluate_duration_probe import EvaluateDurationProbePipeline

    extract(tmp_path / "cache.pt")
    fitted = DurationProbePipeline(
        tmp_path / "fit",
        cache_path=tmp_path / "cache.pt",
        max_steps=2,
        eval_interval=1,
        n_bins=4,
        descriptor_modes=("both",),
        train_shuffled_control=False,
    ).run()
    path = tmp_path / "fit/models/aggregate1_both.pt"

    def evaluate(checkpoint, name):
        return EvaluateDurationProbePipeline(checkpoint, tmp_path / "cache.pt", tmp_path / name).run()

    result = evaluate(path, "evaluation")
    for key, value in result["metrics"]["test"]["model"].items():
        assert value == fitted["aggregation_results"]["1"]["models"]["both"]["metrics"]["test"][key]
    checkpoint = torch.load(path, weights_only=True)
    for key in ("frame_distribution", "training_prior", "speaker_splits"):
        del checkpoint[key]
    torch.save(checkpoint, tmp_path / "legacy.pt")
    legacy = evaluate(tmp_path / "legacy.pt", "legacy_eval")
    assert legacy["legacy_cache_binding"]
    assert legacy["metrics"] == result["metrics"]
    checkpoint["projection"] += 1
    torch.save(checkpoint, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="projection mismatch"):
        evaluate(tmp_path / "wrong.pt", "wrong_eval")


def test_ordered_cache_keeps_complete_sequences_with_bounded_utterance_sampling(tmp_path):
    from quick_convert.pipelines.contextual_duration_probe import validate_sequences

    cache, encoder = extract(
        tmp_path / "seq.pt", store_sequences=True, train_sequence_capacity=2, heldout_sequence_capacity=1
    )
    other, _ = extract(
        tmp_path / "seq_batch.pt",
        batch_size=4,
        store_sequences=True,
        train_sequence_capacity=2,
        heldout_sequence_capacity=1,
    )
    rows = validate_sequences(cache)
    assert encoder.calls == 12
    assert [len(rows[p]) for p in ("train", "validation", "test")] == [2, 1, 1]
    for part in rows:
        for a, b in zip(rows[part], other["sequences"][part], strict=True):
            assert a["utterance"] == b["utterance"]
            torch.testing.assert_close(a["features"], b["features"])
            torch.testing.assert_close(a["frames"], b["frames"])
    # Verify actual ordered means/targets against fresh detector output.
    from quick_convert.components.boundaries import change_signal
    from quick_convert.components.boundaries.measurements import measure_segments

    row = rows["train"][0]
    sample = next(s for s in Dataset().samples if f"{s.split}/{s.utt_id}" == row["utterance"])
    with torch.no_grad():
        values = Encoder().eval()(AudioBatch.from_samples([sample])).values[0]
    detected = measure_segments(
        values,
        change_signal(values, "l2"),
        threshold=0.3,
        min_distance_frames=1,
        frame_hz=50,
        duration_seconds=sample.waveform.numel() / 100,
    )
    interior = [r for r in detected if not r["is_edge"]]
    expected = torch.stack([values[r["start_frame"] : r["end_frame"]].mean(0) for r in interior]) @ cache["projection"]
    torch.testing.assert_close(row["features"], expected)
    assert row["frames"].tolist() == [r["end_frame"] - r["start_frame"] for r in interior]
    skipped, _ = extract(tmp_path / "short.pt", store_sequences=True, max_sequence_segments=1)
    assert all(not x for x in skipped["sequences"].values())
    assert sum(skipped["sequence_sampling"]["skipped_utterances"].values()) == 12
    with pytest.raises(ValueError, match="No retained"):
        validate_sequences(skipped)


def test_context_cnn_padding_cannot_change_valid_predictions():
    from quick_convert.components.boundaries.duration import ContextualDurationCNN
    from quick_convert.training.contextual_duration_probe import predict_sequences

    torch.manual_seed(2)
    model = ContextualDurationCNN(3, 4, hidden_dim=8, dropout=0).eval()
    short, long = torch.randn(3, 3), torch.randn(20, 3)
    center, scale = torch.zeros(3), torch.ones(3)
    alone = predict_sequences(model, [short], center, scale, "cpu")
    batched = predict_sequences(model, [short, long], center, scale, "cpu")[:3]
    torch.testing.assert_close(alone, batched)
    # Batch members cannot provide context to each other.
    changed = predict_sequences(model, [short, long * 100], center, scale, "cpu")[:3]
    torch.testing.assert_close(alone, changed)


def test_cnn_learns_neighbor_information_unavailable_to_independent_head():
    from quick_convert.training.contextual_duration_probe import fit_sequence_probe, predict_sequences

    generator = torch.Generator().manual_seed(17)
    features = {
        part: [torch.randint(2, (24, 1), generator=generator).float() * 2 - 1 for _ in range(20)]
        for part in ("train", "validation")
    }
    # Target depends on preceding segment, not its own mean; no duration is input.
    labels = {part: [(torch.cat((torch.zeros(1), x[:-1, 0])) > 0).long() for x in xs] for part, xs in features.items()}
    predictions = {}
    for architecture in ("independent", "cnn"):
        model, center, scale, _ = fit_sequence_probe(
            features,
            labels,
            torch.ones(2) / 2,
            architecture=architecture,
            hidden_dim=16,
            dropout=0,
            max_steps=160,
            batch_size=4,
            learning_rate=0.01,
            eval_interval=40,
            patience=5,
        )
        probs = predict_sequences(model, features["validation"], center, scale, "cpu")
        predictions[architecture] = float((probs.argmax(-1) == torch.cat(labels["validation"])).float().mean())
    assert predictions["cnn"] > 0.95
    assert predictions["independent"] < 0.7


def test_context_pipeline_cached_reproducible_and_checkpoint_evaluation(tmp_path):
    from quick_convert.pipelines.contextual_duration_probe import ContextualDurationProbePipeline
    from quick_convert.pipelines.evaluate_duration_probe import EvaluateDurationProbePipeline

    cache, encoder = extract(tmp_path / "cache.pt", store_sequences=True)
    kwargs = dict(
        cache_path=tmp_path / "cache.pt",
        max_steps=2,
        eval_interval=1,
        n_bins=4,
        hidden_dim=8,
        train_utterance_batch_size=2,
    )
    first = ContextualDurationProbePipeline(tmp_path / "fit", **kwargs).run()
    second = ContextualDurationProbePipeline(tmp_path / "repeat", **kwargs).run()
    assert first == second
    assert encoder.calls == 12
    assert first["models"]["independent"]["metrics"]["test"]["nll"] == pytest.approx(
        first["models"]["independent"]["shuffled_order_test"]["nll"], abs=1e-6
    )
    for architecture, result in first["models"].items():
        evaluation = EvaluateDurationProbePipeline(
            tmp_path / f"fit/models/sequence_{architecture}.pt",
            tmp_path / "cache.pt",
            tmp_path / f"eval_{architecture}",
        ).run()
        for key, value in evaluation["metrics"]["test"]["model"].items():
            assert (
                value == pytest.approx(result["metrics"]["test"][key])
                if not isinstance(value, dict)
                else value == result["metrics"]["test"][key]
            )
    with pytest.raises(FileExistsError):
        ContextualDurationProbePipeline(tmp_path / "fit", **kwargs).run()
    cache["sequences"]["test"][0]["speaker"] = cache["speaker_splits"]["train"][0]
    torch.save(cache, tmp_path / "leaked.pt")
    with pytest.raises(ValueError, match="speaker assignment"):
        ContextualDurationProbePipeline(tmp_path / "leaked", cache_path=tmp_path / "leaked.pt", max_steps=1).run()
    extract(tmp_path / "old.pt")
    with pytest.raises(ValueError, match="no ordered sequences"):
        ContextualDurationProbePipeline(tmp_path / "old", cache_path=tmp_path / "old.pt", max_steps=1).run()
