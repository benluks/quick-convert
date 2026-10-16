"""Online bounded SSL composition extraction and categorical duration experiments."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from quick_convert.components.boundaries import change_signal
from quick_convert.components.boundaries.duration import (
    composition_moments,
    descriptor_slice,
    duration_labels,
    fit_frame_bins,
)
from quick_convert.components.boundaries.measurements import measure_segments, smooth_changes
from quick_convert.data import AudioBatch
from quick_convert.training.duration_probe import (
    fit_duration_probe,
    macro_gain_interval,
    predict_probabilities,
    probability_metrics,
)


PARTITIONS = ("train", "validation", "test")


def split_speakers(speakers, *, validation_fraction=0.2, test_fraction=0.2, seed=115):
    """Assign speakers before extraction, without consulting duration or features."""
    speakers = sorted(set(speakers))
    if (
        not math.isfinite(validation_fraction)
        or not math.isfinite(test_fraction)
        or len(speakers) < 3
        or validation_fraction <= 0
        or test_fraction <= 0
        or validation_fraction + test_fraction >= 1
    ):
        raise ValueError("Need >=3 speakers and positive held-out fractions totaling <1")
    order = torch.randperm(len(speakers), generator=torch.Generator().manual_seed(seed)).tolist()
    speakers = [speakers[i] for i in order]
    val = max(1, int(len(speakers) * validation_fraction))
    test = max(1, int(len(speakers) * test_fraction))
    if val + test >= len(speakers):
        raise ValueError("Speaker split leaves no training speakers")
    return dict(train=speakers[val + test :], validation=speakers[:val], test=speakers[val : val + test])


class _Examples:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.generator = torch.Generator().manual_seed(seed)
        self.seen = 0
        self.features = None
        self.metadata = []

    def consider(self):
        self.seen += 1
        slot = (
            self.seen - 1
            if self.seen <= self.capacity
            else int(torch.randint(self.seen, (1,), generator=self.generator))
        )
        return slot if slot < self.capacity else None

    def commit(self, slot, features, metadata):
        if self.features is None:
            self.features = torch.empty(self.capacity, *features.shape, dtype=torch.float32)
        self.features[slot] = features.cpu()
        if slot == len(self.metadata):
            self.metadata.append(metadata)
        else:
            self.metadata[slot] = metadata

    def state(self):
        if self.features is None:
            raise ValueError("No eligible examples in a partition/aggregation; broaden the dataset")
        return dict(
            features=self.features[: len(self.metadata)].clone(),
            frames=torch.tensor([m["frames"] for m in self.metadata], dtype=torch.long),
            speakers=[m["speaker"] for m in self.metadata],
            utterances=[m["utterance"] for m in self.metadata],
            eligible_examples=self.seen,
        )


class ExtractDurationCompositionPipeline:
    """One encoder pass; retain only bounded continuous moment sketches.

    Averages/SD use a fixed relative-time grid. A seeded, data-independent random
    projection compresses each moment vector; frame counts remain labels only.
    Reservoirs are separate per speaker partition and aggregation size.
    """

    def __init__(
        self,
        dataset,
        encoder,
        out_path,
        speaker_regex=r"^([^-]+)-",
        aggregation_sizes=(1, 2, 3),
        points=16,
        projection_dim=128,
        train_capacity=20000,
        heldout_capacity=5000,
        validation_fraction=0.2,
        test_fraction=0.2,
        threshold=0.75,
        threshold_mode="utterance_quantile",
        metric="cosine",
        smoothing_frames=1,
        min_distance_ms=20,
        batch_size=1,
        num_workers=0,
        audit_capacity=128,
        seed=115,
    ):
        self.dataset, self.encoder, self.out_path = dataset, encoder, Path(out_path)
        self.speaker_regex = re.compile(speaker_regex)
        if self.speaker_regex.groups != 1:
            raise ValueError("speaker_regex must have exactly one capture group")
        if not aggregation_sizes or any(not isinstance(n, int) or n < 1 for n in aggregation_sizes):
            raise ValueError("Invalid aggregation sizes")
        for name, value in dict(
            points=points,
            projection_dim=projection_dim,
            train_capacity=train_capacity,
            heldout_capacity=heldout_capacity,
            batch_size=batch_size,
            audit_capacity=audit_capacity,
        ).items():
            if not isinstance(value, int) or value < 1 or (name == "points" and value < 2):
                raise ValueError(f"Invalid {name}")
        if (
            threshold_mode not in ("absolute", "utterance_quantile")
            or not math.isfinite(threshold)
            or threshold < 0
            or (threshold_mode == "utterance_quantile" and threshold > 1)
        ):
            raise ValueError("Invalid detector threshold")
        if (
            metric not in ("cosine", "l2")
            or not math.isfinite(min_distance_ms)
            or min_distance_ms < 0
            or num_workers < 0
        ):
            raise ValueError("Invalid detector/loader setting")
        smooth_changes(torch.ones(2), smoothing_frames)
        self.aggregation_sizes = tuple(dict.fromkeys(aggregation_sizes))
        self.points, self.projection_dim = points, projection_dim
        self.train_capacity, self.heldout_capacity = train_capacity, heldout_capacity
        self.validation_fraction, self.test_fraction = validation_fraction, test_fraction
        self.threshold, self.threshold_mode, self.metric = threshold, threshold_mode, metric
        self.smoothing_frames, self.min_distance_ms = smoothing_frames, min_distance_ms
        self.batch_size, self.num_workers, self.audit_capacity, self.seed = (
            batch_size,
            num_workers,
            audit_capacity,
            seed,
        )

    @torch.inference_mode()
    def run(self):
        if self.out_path.exists():
            raise FileExistsError(self.out_path)
        if not hasattr(self.dataset, "rows"):
            raise ValueError("Use a Quick-Convert dataset exposing metadata rows")

        def speaker(utt):
            match = self.speaker_regex.search(utt)
            if match is None:
                raise ValueError(f"speaker_regex does not match {utt!r}")
            return match.group(1)

        splits = split_speakers(
            [speaker(row.utt_id) for row in self.dataset.rows],
            validation_fraction=self.validation_fraction,
            test_fraction=self.test_fraction,
            seed=self.seed,
        )
        assignment = {s: part for part, ids in splits.items() for s in ids}
        reservoirs = {
            size: {
                part: _Examples(
                    self.train_capacity if part == "train" else self.heldout_capacity, self.seed + size * 100 + i
                )
                for i, part in enumerate(PARTITIONS)
            }
            for size in self.aggregation_sizes
        }
        audit = _Examples(self.audit_capacity, self.seed + 9000)
        factors = (0.5, 1.5, 2.0)
        projection = identity = None
        processed = short = 0
        seen = set()
        self.encoder.eval()
        for batch in tqdm(
            self.dataset.make_dataloader(batch_size=self.batch_size, num_workers=self.num_workers),
            desc="Duration composition",
        ):
            for sample in batch:
                key = (sample.split, sample.utt_id)
                if key in seen:
                    raise ValueError(f"Duplicate utterance: {key}")
                seen.add(key)
                spk = speaker(sample.utt_id)
                partition = assignment[spk]
                if sample.waveform is None or sample.sample_rate is None:
                    raise ValueError("Loaded audio is required")
                content = self.encoder(AudioBatch.from_samples([sample]))
                if (
                    content.values.ndim != 3
                    or len(content.values) != 1
                    or content.representation_type != "continuous"
                    or content.temporal_granularity != "frame"
                ):
                    raise ValueError("Use one selected continuous frame encoder layer")
                hz = content.frame_hz
                if hz is None or not math.isfinite(hz) or hz <= 0:
                    raise ValueError("Encoder must report valid frame_hz")
                current = dict(
                    model_name=content.model_name,
                    backend=content.backend,
                    encoder_layer=content.layer,
                    frame_hz=hz,
                    feature_dim=content.feature_dim,
                )
                if identity is not None and current != identity:
                    raise ValueError("Encoder metadata changed")
                identity = current
                processed += 1
                frames = int(content.lengths[0])
                if frames < 2:
                    short += 1
                    del content
                    continue
                values = content.values[0, :frames].detach().cpu().float()
                del content
                if projection is None:
                    dim = values.shape[-1]
                    projection = (
                        torch.eye(dim)
                        if self.projection_dim >= dim
                        else torch.randn(
                            dim, self.projection_dim, generator=torch.Generator().manual_seed(self.seed + 42)
                        )
                        / math.sqrt(self.projection_dim)
                    )
                changes = smooth_changes(change_signal(values, self.metric), self.smoothing_frames)
                cutoff = (
                    float(torch.quantile(changes, self.threshold))
                    if self.threshold_mode == "utterance_quantile"
                    else self.threshold
                )
                rows = measure_segments(
                    values,
                    changes,
                    threshold=cutoff,
                    min_distance_frames=max(1, math.ceil(self.min_distance_ms * hz / 1000 - 1e-9)),
                    frame_hz=hz,
                    duration_seconds=sample.waveform.numel() / int(sample.sample_rate),
                )
                interior = [row for row in rows if not row["is_edge"]]
                pending = []
                for size, parts in reservoirs.items():
                    examples = parts[partition]
                    for start_index in range(len(interior) - size + 1):
                        start = interior[start_index]["start_frame"]
                        end = interior[start_index + size - 1]["end_frame"]
                        if end - start < 2:
                            continue
                        slot = examples.consider()
                        if slot is not None:
                            pending.append((examples, slot, start, end))
                if pending:
                    moments = torch.stack(
                        [composition_moments(values[start:end], self.points) for _, _, start, end in pending]
                    )
                    sketches = moments @ projection
                    for (examples, slot, start, end), sketch in zip(pending, sketches, strict=True):
                        examples.commit(
                            slot,
                            sketch,
                            dict(frames=end - start, speaker=spk, utterance=f"{sample.split}/{sample.utt_id}"),
                        )
                # Audit only single detector segments. Uniform feature-space
                # resampling is a numerical length-cue check, not natural speech.
                for row in interior:
                    start, end = row["start_frame"], row["end_frame"]
                    if end - start < 2:
                        continue
                    slot = audit.consider()
                    if slot is None:
                        continue
                    trajectory = values[start:end]
                    variants = [composition_moments(trajectory, self.points)]
                    for factor in factors:
                        stretched = F.interpolate(
                            trajectory.T[None],
                            size=max(2, round(len(trajectory) * factor)),
                            mode="linear",
                            align_corners=True,
                        )[0].T
                        variants.append(composition_moments(stretched, self.points))
                    audit.commit(
                        slot,
                        torch.stack(variants) @ projection,
                        dict(frames=end - start, speaker=spk, utterance=f"{sample.split}/{sample.utt_id}"),
                    )
                del values, changes
        groups = {
            str(size): {part: examples.state() for part, examples in parts.items()}
            for size, parts in reservoirs.items()
        }
        cache = dict(
            version=1,
            encoder=identity,
            descriptor_points=self.points,
            projection=projection,
            speaker_splits=splits,
            seed=self.seed,
            processed_utterances=processed,
            skipped_short_utterances=short,
            detector=dict(
                threshold=self.threshold,
                threshold_mode=self.threshold_mode,
                metric=self.metric,
                smoothing_frames=self.smoothing_frames,
                min_distance_ms=self.min_distance_ms,
            ),
            groups=groups,
            audit=dict(
                features=audit.features[: len(audit.metadata)].clone() if audit.features is not None else None,
                factors=list(factors),
                metadata=audit.metadata,
            ),
            descriptor="Mean and population SD of raw SSL vectors interpolated onto a fixed relative-time grid, each sketched by a fixed seeded Gaussian projection (identity when requested size >= native dimension). No frame counts/timestamps/path lengths are input features. Aggregates concatenate adjacent interior segments; overlapping aggregates stay within speaker partitions.",
        )
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, self.out_path)
        return self.out_path


class DurationProbePipeline:
    """Orchestrate extraction/cache reuse and categorical probe comparisons."""

    def __init__(
        self,
        out_dir,
        extractor=None,
        cache_path=None,
        n_bins=12,
        descriptor_modes=("mean", "variation", "both"),
        hidden_dim=128,
        dropout=0.1,
        max_steps=1000,
        train_batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-3,
        eval_interval=100,
        patience=5,
        train_shuffled_control=True,
        device="cpu",
        seed=115,
    ):
        if (extractor is None) == (cache_path is None):
            raise ValueError("Provide exactly one of extractor or cache_path")
        for name, value in dict(
            n_bins=n_bins,
            hidden_dim=hidden_dim,
            max_steps=max_steps,
            train_batch_size=train_batch_size,
            eval_interval=eval_interval,
            patience=patience,
        ).items():
            if not isinstance(value, int) or value < 1 or (name == "n_bins" and value < 2):
                raise ValueError(f"Invalid {name}")
        if not descriptor_modes or any(mode not in ("mean", "variation", "both") for mode in descriptor_modes):
            raise ValueError("Invalid descriptor modes")
        if (
            not math.isfinite(dropout)
            or not 0 <= dropout < 1
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
            or not math.isfinite(weight_decay)
            or weight_decay < 0
        ):
            raise ValueError("Invalid optimizer/model setting")
        self.out_dir, self.extractor, self.cache_path = (
            Path(out_dir),
            extractor,
            Path(cache_path) if cache_path is not None else None,
        )
        self.n_bins, self.descriptor_modes = n_bins, tuple(dict.fromkeys(descriptor_modes))
        self.fit_kwargs = dict(
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_steps=max_steps,
            batch_size=train_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            eval_interval=eval_interval,
            patience=patience,
            device=device,
            seed=seed,
        )
        self.train_shuffled_control, self.device, self.seed = train_shuffled_control, device, seed

    def prepare(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if any((self.out_dir / name).exists() for name in ("duration_report.json", "models")):
            raise FileExistsError(f"Duration outputs already exist in {self.out_dir}; choose a new out_dir")

    def write_config(self, rendered_config):
        (self.out_dir / "config.yaml").write_text(rendered_config)

    def run(self):
        self.prepare()
        cache_path = self.extractor.run() if self.extractor is not None else self.cache_path
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache["version"] != 1:
            raise ValueError("Unsupported duration descriptor cache")
        ids = cache["speaker_splits"]
        if any(set(ids[a]) & set(ids[b]) for i, a in enumerate(PARTITIONS) for b in PARTITIONS[i + 1 :]):
            raise ValueError("Speaker leakage in descriptor cache")
        for groups in cache["groups"].values():
            if set(groups) != set(PARTITIONS):
                raise ValueError("Cache must contain train, validation and test partitions")
            if any(
                set(groups[a]["utterances"]) & set(groups[b]["utterances"])
                for i, a in enumerate(PARTITIONS)
                for b in PARTITIONS[i + 1 :]
            ):
                raise ValueError("Utterance leakage in descriptor cache")
            for part, data in groups.items():
                if not set(data["speakers"]) <= set(ids[part]):
                    raise ValueError("Cached example has wrong speaker partition")
                if (
                    len(data["features"]) != len(data["frames"])
                    or len(data["frames"]) != len(data["speakers"])
                    or not len(data["frames"])
                ):
                    raise ValueError("Invalid cached example counts")
                if (
                    data["features"].ndim != 3
                    or data["features"].shape[1:] != (2, cache["projection"].shape[1])
                    or not data["features"].is_floating_point()
                    or not torch.isfinite(data["features"]).all()
                    or data["frames"].ndim != 1
                    or data["frames"].is_floating_point()
                    or torch.any(data["frames"] < 1)
                    or len(data["utterances"]) != len(data["frames"])
                ):
                    raise ValueError("Invalid cached descriptors or integer frame counts")
        results = {}
        (self.out_dir / "models").mkdir()
        generator = torch.Generator().manual_seed(self.seed)
        for size, groups in tqdm(cache["groups"].items(), desc="Categorical duration probes"):
            edges = fit_frame_bins(groups["train"]["frames"], self.n_bins)
            labels = {part: duration_labels(data["frames"], edges) for part, data in groups.items()}
            counts = torch.bincount(labels["train"], minlength=len(edges) + 1).float()
            prior = (counts + 1) / (counts.sum() + len(counts))
            baseline = {
                part: probability_metrics(prior.expand(len(data["frames"]), -1), labels[part], data["speakers"])
                for part, data in groups.items()
            }
            result = dict(
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
                partitions={
                    part: dict(
                        retained_examples=len(data["frames"]),
                        eligible_examples=data["eligible_examples"],
                        speakers=len(set(data["speakers"])),
                        utterances=len(set(data["utterances"])),
                    )
                    for part, data in groups.items()
                },
                unconditional=baseline,
                models={},
            )
            modes = [(mode, False) for mode in self.descriptor_modes]
            if self.train_shuffled_control:
                modes.append(("both", True))
            for mode, shuffled in modes:
                name = mode + ("_shuffled_within_speaker" if shuffled else "")
                print(f"Fitting aggregation={size}, descriptor={name}")
                features = {part: descriptor_slice(data["features"], mode) for part, data in groups.items()}
                fit_labels = {part: labels[part].clone() for part in ("train", "validation")}
                if shuffled:
                    for spk in sorted(set(groups["train"]["speakers"])):
                        indices = torch.tensor([i for i, s in enumerate(groups["train"]["speakers"]) if s == spk])
                        fit_labels["train"][indices] = fit_labels["train"][
                            indices[torch.randperm(len(indices), generator=generator)]
                        ]
                model, center, scale, training = fit_duration_probe(
                    {part: features[part] for part in ("train", "validation")}, fit_labels, prior, **self.fit_kwargs
                )
                metrics = {}
                for part in ("validation", "test"):
                    probabilities = predict_probabilities(model, (features[part] - center) / scale, self.device)
                    scores = probability_metrics(probabilities, labels[part], groups[part]["speakers"])
                    scores["nll_gain_over_unconditional"] = baseline[part]["nll"] - scores["nll"]
                    scores["speaker_macro_nll_gain"] = baseline[part]["speaker_macro_nll"] - scores["speaker_macro_nll"]
                    scores["speaker_bootstrap_95_macro_gain"] = macro_gain_interval(
                        baseline[part], scores, seed=self.seed
                    )
                    metrics[part] = scores
                audit_report = None
                audit = cache["audit"]["features"]
                if size == "1" and audit is not None:
                    original = descriptor_slice(audit[:, 0], mode)
                    original_prob = predict_probabilities(model, (original - center) / scale, self.device)
                    audit_report = []
                    for index, factor in enumerate(cache["audit"]["factors"], 1):
                        altered = descriptor_slice(audit[:, index], mode)
                        relative = (altered - original).norm(dim=-1) / original.norm(dim=-1).clamp_min(1e-8)
                        altered_prob = predict_probabilities(model, (altered - center) / scale, self.device)
                        total_variation = (altered_prob - original_prob).abs().sum(-1) / 2
                        audit_report.append(
                            dict(
                                feature_resampling_factor=factor,
                                count=len(audit),
                                relative_descriptor_change_mean=float(relative.mean()),
                                relative_descriptor_change_p95=float(relative.quantile(0.95)),
                                probability_total_variation_mean=float(total_variation.mean()),
                                probability_total_variation_p95=float(total_variation.quantile(0.95)),
                            )
                        )
                torch.save(
                    dict(
                        version=1,
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
                        descriptor_mode=mode,
                        aggregation_size=int(size),
                        projection=cache["projection"],
                        descriptor_points=cache["descriptor_points"],
                        encoder=cache["encoder"],
                    ),
                    self.out_dir / "models" / f"aggregate{size}_{name}.pt",
                )
                result["models"][name] = dict(training=training, metrics=metrics, resampling_audit=audit_report)
                del model
            results[size] = result
        report = dict(
            version=1,
            cache_path=str(cache_path),
            encoder=cache["encoder"],
            descriptor=cache["descriptor"],
            descriptor_points=cache["descriptor_points"],
            sketch_dimension=cache["projection"].shape[1],
            speaker_splits=ids,
            seed=self.seed,
            detector=cache["detector"],
            aggregation_results=results,
            interpretation="Target is categorical positive frame-count bins, not log duration. Bin edges, Laplace-smoothed unconditional probabilities and feature normalization use training speakers only. Each aggregation size has separate models/bins/baselines. Aggregates overlap within utterances but never across speaker partitions. Early stopping uses validation NLL; test speakers never select checkpoints. Positive NLL gain favors the conditional model. Within-speaker training-label shuffling is a negative control retaining speaker-specific duration marginals. Bootstrap intervals resample held-out speakers. Fixed-grid moments and feature-resampling audits reduce/check numerical length cues but cannot remove timing information intrinsic to SSL features. Descriptor sketches may discard relevant content; negative results are not a rejection of the segmentation or the modeling hypothesis. Probe checkpoints are experimental research outputs, not exported inference systems.",
        )
        (self.out_dir / "duration_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Wrote duration_report.json and probe checkpoints to {self.out_dir}")
        return report
