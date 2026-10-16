"""Online, bounded SSL trajectory retrieval with matched random-location controls."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
from tqdm import tqdm

from quick_convert.components.boundaries import change_signal
from quick_convert.components.boundaries.measurements import measure_segments, smooth_changes
from quick_convert.components.boundaries.trajectories import nearest_trajectories, resample_trajectory, trajectory_dtw
from quick_convert.data import AudioBatch
from quick_convert.utils.streaming_statistics import RunningCovariance, RunningMoments, average_ranks


MOVEMENT = ("mean_internal_l2", "path_length_l2", "net_displacement_l2")


def _summary(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    if not values.numel():
        return dict(count=0, mean=None, p05=None, p50=None, p95=None)
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()
    return dict(count=len(values), mean=float(values.mean()), p05=quantiles[0], p50=quantiles[1], p95=quantiles[2])


def _moments_summary(moments):
    return {k: v if not isinstance(v, float) or math.isfinite(v) else None for k, v in moments.summary().items()}


class _PairedReservoir:
    """Uniform reservoir over detected segments, retaining its matched controls."""

    def __init__(self, capacity, points, generator):
        self.capacity, self.points, self.generator = capacity, points, generator
        self.seen = 0
        self.banks = None
        self.metadata = {"detected": [], "random_location": []}

    def add(self, detected, control, metadata, control_metadata):
        self.seen += 1
        slot = (
            self.seen - 1
            if self.seen <= self.capacity
            else int(torch.randint(self.seen, (1,), generator=self.generator))
        )
        if slot >= self.capacity:
            return
        descriptors = [resample_trajectory(trajectory, self.points).cpu() for trajectory in (detected, control)]
        if self.banks is None:
            self.banks = {name: torch.empty(self.capacity, *descriptors[0].shape) for name in self.metadata}
        for name, descriptor, record in zip(self.metadata, descriptors, (metadata, control_metadata), strict=True):
            self.banks[name][slot] = descriptor
            if slot == len(self.metadata[name]):
                self.metadata[name].append(record)
            else:
                self.metadata[name][slot] = record


class TrajectoryAnalysisPipeline:
    """Match duration-normalized SSL paths, not external content labels.

    One selected encoder layer is required. The reservoir is uniform over
    eligible detected interior segments; each retained control is a random
    contiguous span of exactly the same frame length in the same utterance.
    No corpus embeddings or descriptors are written to disk.
    """

    def __init__(
        self,
        dataset,
        encoder,
        out_dir,
        threshold=0.75,
        threshold_mode="utterance_quantile",
        metric="cosine",
        smoothing_frames=1,
        min_distance_ms=20,
        points=16,
        max_segments=2000,
        max_queries=256,
        top_k=5,
        retrieval_batch_size=32,
        retrieval_device="cpu",
        speaker_regex=None,
        different_speaker=False,
        dtw_band_fraction=0.25,
        batch_size=1,
        num_workers=0,
        max_utterances=None,
        seed=115,
    ):
        if (
            threshold_mode not in ("absolute", "utterance_quantile")
            or not math.isfinite(threshold)
            or threshold < 0
            or (threshold_mode == "utterance_quantile" and threshold > 1)
        ):
            raise ValueError("Invalid threshold or calibration mode")
        if metric not in ("cosine", "l2") or not math.isfinite(min_distance_ms) or min_distance_ms < 0:
            raise ValueError("Invalid change metric or minimum distance")
        smooth_changes(torch.ones(2), smoothing_frames)
        for name, value in dict(
            points=points,
            max_segments=max_segments,
            max_queries=max_queries,
            top_k=top_k,
            retrieval_batch_size=retrieval_batch_size,
            batch_size=batch_size,
        ).items():
            if not isinstance(value, int) or value < 1 or (name in ("points", "max_segments") and value < 2):
                raise ValueError(f"{name} must be a positive integer (points/max_segments >=2)")
        if not math.isfinite(dtw_band_fraction) or not 0 <= dtw_band_fraction <= 1 or num_workers < 0:
            raise ValueError("Invalid DTW band or loader workers")
        if max_utterances is not None and max_utterances < 1:
            raise ValueError("max_utterances must be positive")
        self.speaker_regex = re.compile(speaker_regex) if speaker_regex else None
        if self.speaker_regex is not None and self.speaker_regex.groups != 1:
            raise ValueError("speaker_regex must contain exactly one capture group")
        if different_speaker and self.speaker_regex is None:
            raise ValueError("different_speaker requires speaker_regex")
        self.dataset, self.encoder, self.out_dir = dataset, encoder, Path(out_dir)
        self.threshold, self.threshold_mode, self.metric = threshold, threshold_mode, metric
        self.smoothing_frames, self.min_distance_ms = smoothing_frames, min_distance_ms
        self.points, self.max_segments, self.max_queries, self.top_k = points, max_segments, max_queries, top_k
        self.retrieval_batch_size, self.retrieval_device = retrieval_batch_size, retrieval_device
        self.different_speaker, self.dtw_band_fraction = different_speaker, dtw_band_fraction
        self.batch_size, self.num_workers, self.max_utterances, self.seed = (
            batch_size,
            num_workers,
            max_utterances,
            seed,
        )

    def prepare(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if any((self.out_dir / name).exists() for name in ("trajectory_analysis.json", "trajectory_matches.jsonl")):
            raise FileExistsError(f"Trajectory outputs already exist in {self.out_dir}; choose a fresh out_dir")

    def write_config(self, rendered_config):
        (self.out_dir / "config.yaml").write_text(rendered_config)

    @torch.inference_mode()
    def run(self):
        self.prepare()
        self.encoder.eval()
        generator = torch.Generator().manual_seed(self.seed)
        control_generator = torch.Generator().manual_seed(self.seed + 1)
        reservoir = _PairedReservoir(self.max_segments, self.points, generator)
        control_stats = {
            population: {name: RunningMoments() for name in MOVEMENT}
            for population in ("detected", "random_location", "paired_detected_minus_random")
        }
        control_relations = {
            population: {
                name: {kind: RunningCovariance() for kind in ("pooled", "within_utterance_rank")} for name in MOVEMENT
            }
            for population in ("detected", "random_location")
        }
        loader = self.dataset.make_dataloader(batch_size=self.batch_size, num_workers=self.num_workers)
        processed = short = 0
        identity = None
        seen_utterances = set()
        for batch in tqdm(loader, desc="SSL trajectory sampling"):
            for sample in batch:
                if self.max_utterances is not None and processed >= self.max_utterances:
                    break
                if sample.waveform is None or sample.sample_rate is None:
                    raise ValueError("Trajectory analysis requires loaded audio")
                if sample.waveform.ndim != 1 or not sample.waveform.numel():
                    raise ValueError("Expected nonempty mono audio")
                utt_key = (sample.split, sample.utt_id)
                if utt_key in seen_utterances:
                    raise ValueError(f"Duplicate utterance: {utt_key}")
                seen_utterances.add(utt_key)
                speaker = None
                if self.speaker_regex:
                    match = self.speaker_regex.search(sample.utt_id)
                    if match is None:
                        raise ValueError(f"speaker_regex did not match {sample.utt_id!r}")
                    speaker = match.group(1)
                content = self.encoder(AudioBatch.from_samples([sample]))
                if content.values.ndim != 3 or content.values.shape[0] != 1:
                    raise ValueError("Configure a single encoder layer; expected [1,T,D], not an all-layer output")
                if content.representation_type != "continuous" or content.temporal_granularity != "frame":
                    raise ValueError("Continuous frame representations are required")
                hz = content.frame_hz
                if hz is None or not math.isfinite(hz) or hz <= 0:
                    raise ValueError("Encoder must report a finite positive frame_hz")
                current_identity = dict(
                    model_name=content.model_name, backend=content.backend, encoder_layer=content.layer, frame_hz=hz
                )
                if identity is not None and identity != current_identity:
                    raise ValueError("Encoder representation metadata changed during the run")
                identity = current_identity
                processed += 1
                frames = int(content.lengths[0])
                if frames < 2:
                    short += 1
                    del content
                    continue
                values = content.values[0, :frames].detach().cpu().float()
                del content
                changes = smooth_changes(change_signal(values, self.metric), self.smoothing_frames)
                raw_l2 = change_signal(values, "l2")
                prefix = torch.cat((torch.zeros(1, dtype=torch.float64), raw_l2.double().cumsum(0)))
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
                    internal_l2=raw_l2,
                )
                paired_rows = []
                for row in rows:
                    length = row["end_frame"] - row["start_frame"]
                    if row["is_edge"] or length < 2:
                        continue
                    # Interior durations are exact hop counts; avoid tiny
                    # timestamp-subtraction errors breaking ties in rank controls.
                    row = dict(row, duration_ms=length / hz * 1000)
                    # Both populations exclude recording edges. Random placement
                    # ignores detector peaks and may overlap other control spans.
                    start = int(torch.randint(1, frames - length, (1,), generator=control_generator))
                    end = start + length
                    path = float(prefix[end - 1] - prefix[start])
                    control = dict(
                        start_frame=start,
                        end_frame=end,
                        start_seconds=start / hz,
                        end_seconds=end / hz,
                        duration_ms=length / hz * 1000,
                        mean_internal_l2=path / (length - 1),
                        path_length_l2=path,
                        net_displacement_l2=float(torch.linalg.vector_norm(values[end - 1] - values[start])),
                    )
                    paired_rows.append((row, control))
                    for name in MOVEMENT:
                        control_stats["detected"][name].update([row[name]])
                        control_stats["random_location"][name].update([control[name]])
                        control_stats["paired_detected_minus_random"][name].update([row[name] - control[name]])
                    common = dict(utt_id=sample.utt_id, split=sample.split, path=str(sample.path), speaker_id=speaker)
                    fields = ("start_frame", "end_frame", "start_seconds", "end_seconds", "duration_ms", *MOVEMENT)
                    detected_meta = {**common, **{name: row[name] for name in fields}, "effective_threshold": cutoff}
                    control_meta = {**common, **{name: control[name] for name in fields}, "effective_threshold": cutoff}
                    reservoir.add(
                        values[row["start_frame"] : row["end_frame"]], values[start:end], detected_meta, control_meta
                    )
                for index, population in enumerate(("detected", "random_location")):
                    duration = torch.tensor([pair[index]["duration_ms"] for pair in paired_rows], dtype=torch.float64)
                    for name, states in control_relations[population].items():
                        movement = torch.tensor([pair[index][name] for pair in paired_rows], dtype=torch.float64)
                        states["pooled"].update(movement, duration)
                        if len(duration) >= 3 and duration.max() != duration.min() and movement.max() != movement.min():
                            x, y = average_ranks(movement) / len(movement), average_ranks(duration) / len(duration)
                            states["within_utterance_rank"].update(x - x.mean(), y - y.mean())
                del values, changes, raw_l2, prefix
            if self.max_utterances is not None and processed >= self.max_utterances:
                break
        retained = len(reservoir.metadata["detected"])
        if retained < 2 or len({(m["split"], m["utt_id"]) for m in reservoir.metadata["detected"]}) < 2:
            raise ValueError(
                "Too few eligible segments or distinct utterances for retrieval; broaden the dataset or lower the cutoff"
            )
        queries = torch.randperm(retained, generator=generator)[: self.max_queries].tolist()
        reports = {}
        matches_path = self.out_dir / "trajectory_matches.jsonl"
        with matches_path.open("w") as handle:
            for population in ("detected", "random_location"):
                bank = reservoir.banks[population][:retained]
                metadata = reservoir.metadata[population]
                reports[population] = self._retrieve(bank, metadata, queries, generator, handle, population)
        report = dict(
            version=1,
            encoder=identity,
            processed_utterances=processed,
            skipped_short_utterances=short,
            eligible_detected_segments=reservoir.seen,
            retained_pairs=retained,
            requested_queries=len(queries),
            seed=self.seed,
            descriptor_points=self.points,
            descriptor_storage_bytes=sum(bank.numel() * bank.element_size() for bank in reservoir.banks.values()),
            detector=dict(
                metric=self.metric,
                threshold_mode=self.threshold_mode,
                threshold=self.threshold,
                smoothing_frames=self.smoothing_frames,
                min_distance_ms=self.min_distance_ms,
            ),
            retrieval=dict(
                device=self.retrieval_device,
                top_k=self.top_k,
                different_speaker=self.different_speaker,
                dtw_band_fraction=self.dtw_band_fraction,
            ),
            control_duration_relationships={
                population: {
                    name: {
                        kind: dict(correlation=state.correlation(), pairs=state.x.count)
                        for kind, state in states.items()
                    }
                    for name, states in relations.items()
                }
                for population, relations in control_relations.items()
            },
            paired_control_measurements={
                population: {name: _moments_summary(moments) for name, moments in stats.items()}
                for population, stats in control_stats.items()
            },
            populations=reports,
            interpretation="Uniform bounded reservoir of detected interior segments, each paired with a random-location span from the same utterance with identical frame length. Descriptors preserve SSL location, normalize time to a fixed grid, and normalize each grid point for cosine distance. Duration is excluded from matching. Same utterance is excluded from every candidate pool. DTW reranks only retrieved candidates using retained resampled trajectories. Independent temporal permutations preserve sampled vectors while disrupting order; all controls use the same bank/query counts. Nearest beats random by construction: compare against matched-size null searches, not random pairs alone. Duration ratios among nearest matches do not establish shared content identity or plausible resynthesis. No cross-layer comparison, external alignment, or corpus embedding persistence.",
        )
        (self.out_dir / "trajectory_analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Wrote trajectory_analysis.json and trajectory_matches.jsonl to {self.out_dir}")
        return report

    def _retrieve(self, bank, metadata, queries, generator, handle, population):
        neighbors = nearest_trajectories(
            bank,
            metadata,
            queries,
            top_k=self.top_k,
            batch_size=self.retrieval_batch_size,
            device=self.retrieval_device,
            different_speaker=self.different_speaker,
        )
        permutations = torch.stack([torch.randperm(self.points, generator=generator) for _ in range(len(bank))])
        shuffled = bank[torch.arange(len(bank))[:, None], permutations]
        null_neighbors = nearest_trajectories(
            shuffled,
            metadata,
            queries,
            top_k=1,
            batch_size=self.retrieval_batch_size,
            device=self.retrieval_device,
            different_speaker=self.different_speaker,
        )
        results = {name: [] for name in ("nearest", "dtw_refined", "random_partner", "time_shuffled_nearest")}
        for query_index, candidates, null in tqdm(
            zip(queries, neighbors, null_neighbors, strict=True), total=len(queries), desc=f"Retrieving {population}"
        ):
            if not candidates:
                continue
            query = metadata[query_index]
            eligible = [
                i
                for i, m in enumerate(metadata)
                if (m["split"], m["utt_id"]) != (query["split"], query["utt_id"])
                and (not self.different_speaker or m["speaker_id"] != query["speaker_id"])
            ]
            random_index = eligible[int(torch.randint(len(eligible), (1,), generator=generator))]
            refinements = [
                (index, trajectory_dtw(bank[query_index], bank[index], self.dtw_band_fraction))
                for index, _ in candidates
            ]
            best_index, best_alignment = min(refinements, key=lambda pair: pair[1]["distance"])
            selections = (
                ("nearest", candidates[0][0], candidates[0][1], refinements[0][1]),
                (
                    "dtw_refined",
                    best_index,
                    float((1 - (bank[query_index] * bank[best_index]).sum(-1)).clamp(0, 2).mean()),
                    best_alignment,
                ),
                (
                    "random_partner",
                    random_index,
                    float((1 - (bank[query_index] * bank[random_index]).sum(-1)).clamp(0, 2).mean()),
                    trajectory_dtw(bank[query_index], bank[random_index], self.dtw_band_fraction),
                ),
                ("time_shuffled_nearest", null[0][0], null[0][1], None),
            )
            for condition, index, distance, alignment in selections:
                candidate = metadata[index]
                duration_ratio = max(query["duration_ms"], candidate["duration_ms"]) / min(
                    query["duration_ms"], candidate["duration_ms"]
                )
                record = dict(
                    population=population,
                    condition=condition,
                    query=query,
                    candidate=candidate,
                    cosine_distance=distance,
                    duration_ratio=duration_ratio,
                    same_speaker=query["speaker_id"] == candidate["speaker_id"] if self.speaker_regex else None,
                    dtw=alignment,
                    eligible_candidates=len(eligible),
                )
                results[condition].append(record)
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        summary = {}
        for condition, rows in results.items():
            by_similarity = sorted(rows, key=lambda row: row["cosine_distance"])
            summary[condition] = dict(
                queries_with_matches=len(rows),
                cosine_distance=_summary([r["cosine_distance"] for r in rows]),
                duration_ratio=_summary([r["duration_ratio"] for r in rows]),
                fraction_duration_ratio_ge_1_5=sum(r["duration_ratio"] >= 1.5 for r in rows) / len(rows)
                if rows
                else None,
                fraction_duration_ratio_ge_2=sum(r["duration_ratio"] >= 2 for r in rows) / len(rows) if rows else None,
                same_speaker_fraction=sum(r["same_speaker"] for r in rows) / len(rows)
                if rows and self.speaker_regex
                else None,
                dtw_distance=_summary([r["dtw"]["distance"] for r in rows if r["dtw"] is not None]),
                dtw_cost_reduction=_summary(
                    [r["cosine_distance"] - r["dtw"]["distance"] for r in rows if r["dtw"] is not None]
                ),
                non_diagonal_step_fraction=_summary(
                    [r["dtw"]["non_diagonal_step_fraction"] for r in rows if r["dtw"] is not None]
                ),
                similarity_quartiles=[
                    dict(
                        quartile=i + 1,
                        cosine_distance=_summary(
                            [r["cosine_distance"] for r in by_similarity[i * len(rows) // 4 : (i + 1) * len(rows) // 4]]
                        ),
                        duration_ratio=_summary(
                            [r["duration_ratio"] for r in by_similarity[i * len(rows) // 4 : (i + 1) * len(rows) // 4]]
                        ),
                    )
                    for i in range(4)
                ],
            )
        speakers = {m["speaker_id"] for m in metadata if m["speaker_id"] is not None}
        return dict(
            retained_utterances=len({(m["split"], m["utt_id"]) for m in metadata}),
            retained_speakers=len(speakers) if self.speaker_regex else None,
            conditions=summary,
        )
