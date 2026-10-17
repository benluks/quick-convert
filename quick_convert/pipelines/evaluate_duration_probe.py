"""Score saved research duration heads without fitting or running an SSL encoder."""

import json
from pathlib import Path

import torch

from quick_convert.components.boundaries.duration import CategoricalDurationProbe, descriptor_slice, duration_labels
from quick_convert.training.duration_probe import (
    duration_error_metrics,
    fit_frame_distribution,
    predict_probabilities,
    probability_metrics,
)


class EvaluateDurationProbePipeline:
    """Evaluate one checkpoint against its matching original-frame descriptor cache.

    Older checkpoints reconstruct within-bin distributions from cache training
    durations only. Held-out durations are used exclusively as scoring targets.
    """

    def __init__(self, checkpoint_path, cache_path, out_dir, device="cpu"):
        self.checkpoint_path, self.cache_path = Path(checkpoint_path), Path(cache_path)
        self.out_dir, self.device = Path(out_dir), device

    def prepare(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if (self.out_dir / "duration_evaluation.json").exists():
            raise FileExistsError("Choose a new evaluation out_dir")

    def write_config(self, rendered_config):
        (self.out_dir / "config.yaml").write_text(rendered_config)

    def run(self):
        self.prepare()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        cache = torch.load(self.cache_path, map_location="cpu", weights_only=True)
        if checkpoint["version"] != 1 or cache["version"] != 2:
            raise ValueError("Unsupported duration checkpoint/cache version")
        for key in ("encoder", "descriptor_method"):
            if checkpoint[key] != cache[key]:
                raise ValueError(f"Checkpoint/cache mismatch: {key}")
        if checkpoint["descriptor_method"] != "original_frame_moments":
            raise ValueError("Expected original-frame descriptors")
        if not torch.equal(checkpoint["projection"], cache["projection"]):
            raise ValueError("Checkpoint/cache projection mismatch")
        splits = cache["speaker_splits"]
        if any(set(splits[a]) & set(splits[b]) for i, a in enumerate(splits) for b in list(splits)[i + 1 :]):
            raise ValueError("Speaker leakage in cache")
        if "speaker_splits" in checkpoint and checkpoint["speaker_splits"] != splits:
            raise ValueError("Checkpoint/cache speaker split mismatch")
        groups = cache["groups"][str(checkpoint["aggregation_size"])]
        for part, data in groups.items():
            if not set(data["speakers"]) <= set(splits[part]):
                raise ValueError("Cached examples disagree with speaker splits")
        edges = checkpoint["bin_upper_edges_frames"]
        train = groups["train"]["frames"]
        counts = torch.bincount(duration_labels(train, edges), minlength=len(edges) + 1).float()
        prior = (counts + 1) / (counts.sum() + len(counts))
        distribution = fit_frame_distribution(train, edges)
        # New checkpoints bind evaluation to the same training duration statistics.
        if "frame_distribution" in checkpoint:
            if any(
                not torch.equal(value, distribution[key]) for key, value in checkpoint["frame_distribution"].items()
            ):
                raise ValueError("Checkpoint/cache training duration distribution mismatch")
            if not torch.equal(prior, checkpoint["training_prior"]):
                raise ValueError("Checkpoint/cache training prior mismatch")
        model = CategoricalDurationProbe(**checkpoint["model_config"]).to(self.device)
        model.load_state_dict(checkpoint["state_dict"])
        results = {}
        for part in ("validation", "test"):
            data = groups[part]
            features = descriptor_slice(data["features"], checkpoint["descriptor_mode"])
            normalized = (features - checkpoint["normalization_mean"]) / checkpoint["normalization_scale"]
            probabilities = predict_probabilities(model, normalized, self.device)
            labels = duration_labels(data["frames"], edges)
            results[part] = {}
            for name, probs in (("model", probabilities), ("unconditional", prior.expand(len(labels), -1))):
                scores = probability_metrics(probs, labels, data["speakers"])
                scores.update(
                    duration_error_metrics(probs, data["frames"], edges, distribution, cache["encoder"]["frame_hz"])
                )
                results[part][name] = scores
        report = dict(
            version=1,
            checkpoint_path=str(self.checkpoint_path),
            cache_path=str(self.cache_path),
            aggregation_size=checkpoint["aggregation_size"],
            bin_upper_edges_frames=edges.tolist(),
            legacy_cache_binding="speaker_splits" not in checkpoint,
            metrics=results,
            interpretation="No training or checkpoint selection performed. Interval error uses the modal bin and an unbounded last interval. Point MAE uses the median of the probability mixture of training-only empirical within-bin frame distributions. RPS is normalized by the number of bin boundaries. Older checkpoints cannot verify the original training speaker assignment; supply the original matching cache.",
        )
        (self.out_dir / "duration_evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
