"""Contextual extension of the duration experiment using ordered segment means."""

import json
import math
from pathlib import Path

import torch

from quick_convert.components.boundaries.duration import (
    CategoricalDurationProbe,
    ContextualDurationCNN,
    duration_labels,
    fit_frame_bins,
)
from quick_convert.training.contextual_duration_probe import fit_sequence_probe, predict_sequences
from quick_convert.training.duration_probe import (
    duration_error_metrics,
    fit_frame_distribution,
    macro_gain_interval,
    probability_metrics,
)


PARTITIONS = ("train", "validation", "test")


def validate_sequences(cache):
    """Reject missing sequences, invalid descriptors and partition leakage."""
    if cache["version"] != 2 or cache.get("descriptor_method") != "original_frame_moments":
        raise ValueError("Expected version 2 original-frame descriptor cache")
    sequences = cache.get("sequences")
    if not sequences or set(sequences) != set(PARTITIONS):
        raise ValueError("Cache has no ordered sequences; rerun extraction with store_sequences=true")
    splits = cache["speaker_splits"]
    if set(splits) != set(PARTITIONS) or any(
        set(splits[a]) & set(splits[b]) for i, a in enumerate(PARTITIONS) for b in PARTITIONS[i + 1 :]
    ):
        raise ValueError("Speaker leakage in sequence cache")
    seen = set()
    for part in PARTITIONS:
        if not sequences[part]:
            raise ValueError(f"No retained utterances in {part}")
        for row in sequences[part]:
            x, frames = row["features"], row["frames"]
            if row["utterance"] in seen:
                raise ValueError("Duplicate utterance or utterance leakage in sequence cache")
            seen.add(row["utterance"])
            if (
                row["speaker"] not in splits[part]
                or x.ndim != 2
                or not len(x)
                or x.shape[1] != cache["projection"].shape[1]
                or not x.is_floating_point()
                or not torch.isfinite(x).all()
                or frames.ndim != 1
                or len(frames) != len(x)
                or frames.dtype != torch.long
                or (frames < 2).any()
            ):
                raise ValueError("Invalid ordered sequence or speaker assignment")
    return sequences


def partition_targets(rows, edges):
    frames = torch.cat([row["frames"] for row in rows])
    speakers = [row["speaker"] for row in rows for _ in row["frames"]]
    return frames, duration_labels(frames, edges), speakers


def score_sequence_probabilities(probabilities, rows, edges, distribution, frame_hz):
    frames, labels, speakers = partition_targets(rows, edges)
    scores = probability_metrics(probabilities, labels, speakers)
    scores.update(duration_error_metrics(probabilities, frames, edges, distribution, frame_hz))
    return scores


def evaluate_sequence_checkpoint(checkpoint, cache, device):
    """Score saved independent/CNN sequence heads with no training."""
    if checkpoint["version"] != 1:
        raise ValueError("Unsupported sequence checkpoint version")
    sequences = validate_sequences(cache)
    for key in ("encoder", "descriptor_method", "speaker_splits"):
        if checkpoint[key] != cache[key]:
            raise ValueError(f"Checkpoint/cache mismatch: {key}")
    if not torch.equal(checkpoint["projection"], cache["projection"]):
        raise ValueError("Checkpoint/cache projection mismatch")
    edges = checkpoint["bin_upper_edges_frames"]
    frames = torch.cat([row["frames"] for row in sequences["train"]])
    distribution = fit_frame_distribution(frames, edges)
    if any(not torch.equal(value, distribution[key]) for key, value in checkpoint["frame_distribution"].items()):
        raise ValueError("Checkpoint/cache training duration distribution mismatch")
    counts = torch.bincount(duration_labels(frames, edges), minlength=len(edges) + 1).float()
    prior = (counts + 1) / (counts.sum() + len(counts))
    if not torch.equal(prior, checkpoint["training_prior"]):
        raise ValueError("Checkpoint/cache training prior mismatch")
    constructor = {"independent": CategoricalDurationProbe, "cnn": ContextualDurationCNN}[checkpoint["architecture"]]
    model = constructor(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    results = {}
    for part in ("validation", "test"):
        rows = sequences[part]
        probabilities = predict_sequences(
            model,
            [r["features"] for r in rows],
            checkpoint["normalization_mean"],
            checkpoint["normalization_scale"],
            device,
        )
        count = sum(len(r["frames"]) for r in rows)
        results[part] = {
            name: score_sequence_probabilities(p, rows, edges, distribution, cache["encoder"]["frame_hz"])
            for name, p in (("model", probabilities), ("unconditional", prior.expand(count, -1)))
        }
    return dict(
        version=1,
        probe_type="segment_sequence",
        architecture=checkpoint["architecture"],
        metrics=results,
        bin_upper_edges_frames=edges.tolist(),
    )


class ContextualDurationProbePipeline:
    """Compare an independent MLP and bidirectional CNN on identical utterances.

    Only projected segment means enter the models. Targets, metadata and padding
    masks are separate. Fit statistics and bins use training speakers only.
    """

    def __init__(
        self,
        out_dir,
        extractor=None,
        cache_path=None,
        n_bins=12,
        hidden_dim=128,
        dropout=0.1,
        max_steps=1000,
        train_utterance_batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-3,
        eval_interval=100,
        patience=5,
        device="cpu",
        seed=115,
    ):
        if (extractor is None) == (cache_path is None):
            raise ValueError("Provide exactly one of extractor or cache_path")
        for value in (n_bins, hidden_dim, max_steps, train_utterance_batch_size, eval_interval, patience):
            if not isinstance(value, int) or value < 1:
                raise ValueError("Model/training counts must be positive integers")
        if (
            n_bins < 2
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
            or not math.isfinite(weight_decay)
            or weight_decay < 0
        ):
            raise ValueError("Invalid bins/dropout/optimizer setting")
        self.out_dir, self.extractor = Path(out_dir), extractor
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.n_bins, self.device, self.seed = n_bins, device, seed
        self.fit_kwargs = dict(
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_steps=max_steps,
            batch_size=train_utterance_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            eval_interval=eval_interval,
            patience=patience,
            device=device,
            seed=seed,
        )

    def prepare(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if any((self.out_dir / name).exists() for name in ("contextual_duration_report.json", "models")):
            raise FileExistsError("Choose a new contextual duration out_dir")

    def write_config(self, rendered_config):
        (self.out_dir / "config.yaml").write_text(rendered_config)

    def run(self):
        self.prepare()
        cache_path = self.extractor.run() if self.extractor is not None else self.cache_path
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        sequences = validate_sequences(cache)
        features = {part: [row["features"] for row in rows] for part, rows in sequences.items()}
        train_frames = torch.cat([row["frames"] for row in sequences["train"]])
        edges = fit_frame_bins(train_frames, self.n_bins)
        labels = {part: [duration_labels(row["frames"], edges) for row in rows] for part, rows in sequences.items()}
        counts = torch.bincount(torch.cat(labels["train"]), minlength=len(edges) + 1).float()
        prior = (counts + 1) / (counts.sum() + len(counts))
        distribution = fit_frame_distribution(train_frames, edges)
        baseline = {
            part: score_sequence_probabilities(
                prior.expand(sum(len(row["frames"]) for row in rows), -1),
                rows,
                edges,
                distribution,
                cache["encoder"]["frame_hz"],
            )
            for part, rows in sequences.items()
        }
        (self.out_dir / "models").mkdir()
        models = {}
        for architecture in ("independent", "cnn"):
            print(f"Fitting sequence duration architecture={architecture}")
            model, center, scale, training = fit_sequence_probe(
                {p: features[p] for p in ("train", "validation")},
                {p: labels[p] for p in ("train", "validation")},
                prior,
                architecture=architecture,
                **self.fit_kwargs,
            )
            metrics = {}
            for part in ("validation", "test"):
                probs = predict_sequences(model, features[part], center, scale, self.device)
                scores = score_sequence_probabilities(
                    probs, sequences[part], edges, distribution, cache["encoder"]["frame_hz"]
                )
                scores["nll_gain_over_unconditional"] = baseline[part]["nll"] - scores["nll"]
                scores["speaker_bootstrap_95_macro_gain"] = macro_gain_interval(baseline[part], scores, seed=self.seed)
                metrics[part] = scores
            # Same segment vectors/targets, but order is permuted inside each test
            # utterance. Restore prediction order before scoring original targets.
            generator = torch.Generator().manual_seed(self.seed + 14000)
            permutations = [torch.randperm(len(x), generator=generator) for x in features["test"]]
            shuffled = [x[index] for x, index in zip(features["test"], permutations, strict=True)]
            shuffled_probs = predict_sequences(model, shuffled, center, scale, self.device)
            restored = torch.cat(
                [
                    p[index.argsort()]
                    for p, index in zip(shuffled_probs.split([len(x) for x in shuffled]), permutations, strict=True)
                ]
            )
            shuffled_scores = score_sequence_probabilities(
                restored, sequences["test"], edges, distribution, cache["encoder"]["frame_hz"]
            )
            checkpoint = dict(
                version=1,
                probe_type="segment_sequence",
                architecture=architecture,
                model_config=dict(
                    input_dim=len(center),
                    n_bins=len(prior),
                    hidden_dim=self.fit_kwargs["hidden_dim"],
                    dropout=self.fit_kwargs["dropout"],
                ),
                state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                normalization_mean=center,
                normalization_scale=scale,
                bin_upper_edges_frames=edges,
                frame_distribution=distribution,
                training_prior=prior,
                speaker_splits=cache["speaker_splits"],
                encoder=cache["encoder"],
                descriptor_method=cache["descriptor_method"],
                projection=cache["projection"],
            )
            torch.save(checkpoint, self.out_dir / "models" / f"sequence_{architecture}.pt")
            models[architecture] = dict(
                training=training,
                metrics=metrics,
                shuffled_order_test=shuffled_scores,
                parameters=sum(p.numel() for p in model.parameters()),
            )
        comparison = {
            part: dict(
                nll_gain_cnn_over_independent=models["independent"]["metrics"][part]["nll"]
                - models["cnn"]["metrics"][part]["nll"],
                speaker_bootstrap_95_macro_gain=macro_gain_interval(
                    models["independent"]["metrics"][part], models["cnn"]["metrics"][part], seed=self.seed
                ),
            )
            for part in ("validation", "test")
        }
        report = dict(
            version=1,
            cache_path=str(cache_path),
            seed=self.seed,
            training_settings=self.fit_kwargs,
            encoder=cache["encoder"],
            speaker_splits=cache["speaker_splits"],
            sequence_sampling=cache["sequence_sampling"],
            partitions={
                part: dict(
                    utterances=len(rows),
                    segments=sum(len(row["frames"]) for row in rows),
                    speakers=len({r["speaker"] for r in rows}),
                )
                for part, rows in sequences.items()
            },
            bin_upper_edges_frames=edges.tolist(),
            bins=[
                dict(
                    lower_frames=1 if i == 0 else int(edges[i - 1]) + 1,
                    upper_frames=int(edges[i]) if i < len(edges) else None,
                    training_count=int(counts[i]),
                    training_probability=float(prior[i]),
                )
                for i in range(len(counts))
            ],
            unconditional=baseline,
            models=models,
            comparison=comparison,
            interpretation="Both models use identical projected original-frame segment means, training normalization, speaker splits, duration bins, utterance sampling and segment-weighted loss. CNN is bidirectional, with 15-segment receptive field; it sees no target durations or original frame positions. WavLM features already include utterance context. Padding is masked after every CNN layer. Whole interior sequences are retained without cropping; recording-edge segments are excluded. Test shuffled-order diagnostic permutes vectors within utterances then restores prediction order; it preserves each segment input but changes its neighbors. CNN has more parameters than the MLP, so improved performance alone does not isolate context from capacity. No test information selects checkpoints. Independent per-position output distributions are not a joint generative duration model.",
        )
        (self.out_dir / "contextual_duration_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(f"Wrote contextual_duration_report.json and checkpoints to {self.out_dir}")
        return report
