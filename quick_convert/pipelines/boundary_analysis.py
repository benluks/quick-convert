"""Online boundary experiments: retain measurements and bounded statistics, never embeddings."""

from __future__ import annotations

import csv
import json
import math
from contextlib import ExitStack
from itertools import product
from pathlib import Path

import torch
from tqdm import tqdm

from quick_convert.components.boundaries import change_signal
from quick_convert.components.boundaries.measurements import measure_segments, smooth_changes
from quick_convert.data import AudioBatch
from quick_convert.utils.streaming_statistics import PrioritySample, RunningMoments


MEASURES = (
    "duration_ms",
    "left_strength",
    "right_strength",
    "mean_internal_l2",
    "path_length_l2",
    "net_displacement_l2",
    "prev_duration_ms",
    "next_duration_ms",
)


class BoundaryAnalysisPipeline:
    """Sweep one encoder online with memory bounded independently of corpus size.

    Encoder calls always contain exactly one unpadded utterance. ``batch_size``
    only controls loading, so changing it cannot introduce encoder padding
    effects. All requested layers share that utterance's single forward pass.
    Quantile thresholds are calibrated separately per utterance/layer/metric/
    smoothing setting; they are not corpus-calibrated or shared across models.
    """

    def __init__(
        self,
        dataset,
        encoder,
        out_dir,
        layers=None,
        metrics=("cosine",),
        thresholds=(0.5, 0.75, 0.9, 0.95),
        threshold_mode="utterance_quantile",
        smoothing_frames=(1,),
        min_distance_ms=(20, 40, 80),
        batch_size=1,
        num_workers=0,
        quantile_sample_size=10000,
        seed=0,
        write_segments=True,
        max_utterances=None,
    ):
        if threshold_mode not in {"absolute", "utterance_quantile"}:
            raise ValueError("threshold_mode must be absolute or utterance_quantile")
        if not metrics or any(metric not in {"cosine", "l2"} for metric in metrics):
            raise ValueError("metrics must contain cosine and/or l2")
        if not thresholds or any(not math.isfinite(x) or x < 0 for x in thresholds):
            raise ValueError("thresholds must be finite and nonnegative")
        if threshold_mode == "utterance_quantile" and any(x > 1 for x in thresholds):
            raise ValueError("Quantiles must be in [0, 1]")
        if not smoothing_frames:
            raise ValueError("smoothing_frames cannot be empty")
        for window in smoothing_frames:
            smooth_changes(torch.ones(2), window)
        if not min_distance_ms or any(not math.isfinite(x) or x < 0 for x in min_distance_ms):
            raise ValueError("Minimum distances must be finite and nonnegative")
        if batch_size < 1 or num_workers < 0 or quantile_sample_size < 1:
            raise ValueError("Invalid loader or quantile sample size")
        if max_utterances is not None and max_utterances < 1:
            raise ValueError("max_utterances must be positive")
        if layers is not None and (not layers or any(not isinstance(x, int) for x in layers)):
            raise ValueError("layers must be a nonempty list of integer output-axis indices")
        self.dataset, self.encoder, self.out_dir = dataset, encoder, Path(out_dir)
        self.layers = None if layers is None else list(dict.fromkeys(layers))
        self.metrics, self.thresholds = list(dict.fromkeys(metrics)), list(dict.fromkeys(thresholds))
        self.threshold_mode = threshold_mode
        self.smoothing_frames = list(dict.fromkeys(smoothing_frames))
        self.min_distance_ms = list(dict.fromkeys(min_distance_ms))
        self.batch_size, self.num_workers = batch_size, num_workers
        self.quantile_sample_size, self.seed = quantile_sample_size, seed
        self.write_segments, self.max_utterances = write_segments, max_utterances

    def prepare(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Do not silently overwrite partial results or mix different experiments.
        if any((self.out_dir / name).exists() for name in ("utterances.jsonl", "summary.json", "segments.jsonl")):
            raise FileExistsError(f"Boundary outputs already exist in {self.out_dir}; choose a new out_dir")

    def write_config(self, rendered_config):
        (self.out_dir / "config.yaml").write_text(rendered_config)

    @torch.inference_mode()
    def run(self):
        self.prepare()
        loader = self.dataset.make_dataloader(batch_size=self.batch_size, num_workers=self.num_workers)
        groups, processed, skipped = {}, 0, 0
        self.encoder.eval()
        generator = torch.Generator().manual_seed(self.seed)
        with ExitStack() as stack:
            utterances = stack.enter_context((self.out_dir / "utterances.jsonl").open("w"))
            segments = stack.enter_context((self.out_dir / "segments.jsonl").open("w")) if self.write_segments else None
            for batch in tqdm(loader, desc="Boundary analysis"):
                for sample in batch:
                    if self.max_utterances is not None and processed >= self.max_utterances:
                        break
                    if sample.waveform is None or sample.sample_rate is None:
                        raise ValueError("Boundary analysis requires loaded audio (dataset.load=true)")
                    if sample.waveform.ndim != 1 or sample.waveform.numel() == 0:
                        raise ValueError("Expected nonempty mono audio")
                    content = self.encoder(AudioBatch.from_samples([sample]))
                    if content.values.shape[0] != 1 or content.values.ndim not in (3, 4):
                        raise ValueError("Expected encoder output [1,T,D] or [1,T,L,D]")
                    if content.representation_type != "continuous" or content.temporal_granularity != "frame":
                        raise ValueError("Boundary analysis requires continuous frame representations")
                    hz = content.frame_hz
                    if hz is None or not math.isfinite(hz) or hz <= 0:
                        raise ValueError("Encoder must report a finite positive frame_hz")
                    length = int(content.lengths[0])
                    processed += 1
                    metadata = dict(
                        utt_id=sample.utt_id,
                        split=sample.split,
                        path=str(sample.path),
                        model_name=content.model_name,
                        backend=content.backend,
                        encoder_layer=content.layer,
                        frame_hz=hz,
                        num_frames=length,
                        duration_seconds=sample.waveform.numel() / int(sample.sample_rate),
                    )
                    if length < 2:
                        skipped += 1
                        utterances.write(json.dumps(dict(**metadata, status="too_few_frames")) + "\n")
                        del content
                        continue
                    values = content.values[0, :length].detach().cpu().float()
                    if values.ndim == 2:
                        values = values[:, None, :]
                    count_layers = values.shape[1]
                    selected = list(range(count_layers)) if self.layers is None else self.layers
                    resolved = [index if index >= 0 else count_layers + index for index in selected]
                    if any(index < 0 or index >= count_layers for index in resolved) or len(set(resolved)) != len(
                        resolved
                    ):
                        raise ValueError(f"layers must select distinct valid output-axis indices (size {count_layers})")
                    for layer_index in resolved:
                        embedding = values[:, layer_index]
                        internal_l2 = change_signal(embedding, "l2")
                        for metric, window in product(self.metrics, self.smoothing_frames):
                            changes = smooth_changes(change_signal(embedding, metric), window)
                            for threshold, distance in product(self.thresholds, self.min_distance_ms):
                                cutoff = (
                                    float(torch.quantile(changes, threshold))
                                    if self.threshold_mode == "utterance_quantile"
                                    else threshold
                                )
                                distance_frames = max(1, math.ceil(distance * hz / 1000 - 1e-9))
                                setting = dict(
                                    layer_index=layer_index,
                                    metric=metric,
                                    smoothing_frames=window,
                                    threshold_mode=self.threshold_mode,
                                    threshold=threshold,
                                    min_distance_ms=distance,
                                    min_distance_frames=distance_frames,
                                )
                                rows = measure_segments(
                                    embedding,
                                    changes,
                                    threshold=cutoff,
                                    min_distance_frames=distance_frames,
                                    frame_hz=hz,
                                    duration_seconds=metadata["duration_seconds"],
                                    internal_l2=internal_l2,
                                )
                                utterances.write(
                                    json.dumps(
                                        dict(
                                            **metadata,
                                            **setting,
                                            effective_threshold=cutoff,
                                            status="ok",
                                            boundaries=len(rows) - 1,
                                            segments=len(rows),
                                        )
                                    )
                                    + "\n"
                                )
                                for row in rows:
                                    if segments is not None:
                                        segments.write(
                                            json.dumps(dict(**metadata, **setting, effective_threshold=cutoff, **row))
                                            + "\n"
                                        )
                                    # Keep edge and interior populations separate, including empty interior groups.
                                for population in ("all", "interior"):
                                    key = (
                                        sample.split,
                                        content.model_name,
                                        content.backend,
                                        str(content.layer),
                                        hz,
                                        layer_index,
                                        metric,
                                        window,
                                        threshold,
                                        distance,
                                        population,
                                    )
                                    if key not in groups:
                                        groups[key] = dict(
                                            metadata=dict(
                                                split=sample.split,
                                                model_name=content.model_name,
                                                backend=content.backend,
                                                encoder_layer=content.layer,
                                                frame_hz=hz,
                                                **setting,
                                                population=population,
                                            ),
                                            utterances=0,
                                            boundaries=0,
                                            seconds=0.0,
                                            moments={name: RunningMoments() for name in MEASURES},
                                            sample=PrioritySample(self.quantile_sample_size, generator),
                                        )
                                    group = groups[key]
                                    group["utterances"] += 1
                                    group["boundaries"] += len(rows) - 1
                                    group["seconds"] += metadata["duration_seconds"]
                                    chosen = (
                                        rows if population == "all" else [row for row in rows if not row["is_edge"]]
                                    )
                                    for name in MEASURES:
                                        group["moments"][name].update(
                                            [row[name] for row in chosen if row[name] is not None]
                                        )
                                    group["sample"].update(
                                        torch.tensor([row["duration_ms"] for row in chosen], dtype=torch.float64)
                                    )
                    # Drop GPU output before the next forward pass (including Python view references).
                    del content, values, embedding, changes, internal_l2
                if self.max_utterances is not None and processed >= self.max_utterances:
                    break
        summaries = []
        for group in groups.values():
            sample = group["sample"].values
            quantiles = (
                torch.quantile(sample, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64)).tolist()
                if len(sample)
                else [None] * 5
            )
            summary = dict(
                **group["metadata"],
                utterances=group["utterances"],
                boundaries=group["boundaries"],
                audio_seconds=group["seconds"],
                boundaries_per_second=group["boundaries"] / group["seconds"],
                duration_quantile_sample_count=len(sample),
            )
            for name, moments in group["moments"].items():
                summary.update(
                    {
                        f"{name}_{key}": value if not isinstance(value, float) or math.isfinite(value) else None
                        for key, value in moments.summary().items()
                    }
                )
            summary.update(
                dict(
                    zip(
                        ("duration_ms_p05", "duration_ms_p25", "duration_ms_p50", "duration_ms_p75", "duration_ms_p95"),
                        quantiles,
                        strict=True,
                    )
                )
            )
            summaries.append(summary)
        result = dict(
            version=1,
            processed_utterances=processed,
            skipped_short_utterances=skipped,
            encoder_call_batch_size=1,
            seed=self.seed,
            quantile_sample_size=self.quantile_sample_size,
            timing="internal boundaries at frame_index/frame_hz; last endpoint is audio duration",
            quantiles="bounded uniform priority sample; means and population standard deviations are exact streaming moments",
            groups=summaries,
        )
        (self.out_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if summaries:
            with (self.out_dir / "summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)
        return result
