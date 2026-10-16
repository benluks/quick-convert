"""Streaming post-analysis of boundary records; no encoder or audio dependency."""

from __future__ import annotations

import csv
import json
import math
import re
from itertools import groupby
from pathlib import Path

import torch
from tqdm import tqdm

from quick_convert.utils.streaming_statistics import RunningMoments


SETTING_FIELDS = (
    "split",
    "model_name",
    "backend",
    "encoder_layer",
    "frame_hz",
    "layer_index",
    "metric",
    "smoothing_frames",
    "threshold_mode",
    "threshold",
    "min_distance_ms",
    "min_distance_frames",
)
RELATIONSHIPS = {
    "preceding_strength": "left_strength",
    "following_strength": "right_strength",
    "internal_movement": "mean_internal_l2",
    "next_duration": "next_duration_ms",
}


def average_ranks(values):
    """One-based ranks with average ties, computed only within one utterance."""
    _, inverse, counts = torch.unique(values, sorted=True, return_inverse=True, return_counts=True)
    ranks = counts.cumsum(0).double() - (counts.double() - 1) / 2
    return ranks[inverse]


class _Pairs:
    """Stable merge of paired moments, with optional within-utterance ranking."""

    def __init__(self):
        self.x, self.y = RunningMoments(), RunningMoments()
        self.cross = 0.0

    def update(self, x, y):
        if not len(x):
            return
        dx, dy = x - x.mean(), y - y.mean()
        old, n = self.x.count, len(x)
        self.cross += float((dx * dy).sum()) + (float(x.mean()) - self.x.mean) * (
            float(y.mean()) - self.y.mean
        ) * old * n / (old + n)
        self.x.update(x)
        self.y.update(y)

    def correlation(self):
        denominator = math.sqrt(max(self.x.m2, 0) * max(self.y.m2, 0))
        return max(-1.0, min(1.0, self.cross / denominator)) if denominator else None


class SummarizeBoundariesPipeline:
    """Read contiguous segmentation records and write small aggregate reports.

    The input is the unmodified segments.jsonl emitted by BoundaryAnalysisPipeline:
    each utterance/setting's segments must remain contiguous and time ordered.
    Memory holds one segmentation plus aggregate per-setting/per-speaker state.
    """

    def __init__(self, segments_path, out_dir, speaker_regex=None, bootstrap_samples=1000, seed=115):
        if bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be nonnegative")
        self.segments_path, self.out_dir = Path(segments_path), Path(out_dir)
        self.speaker_regex = re.compile(speaker_regex) if speaker_regex else None
        if self.speaker_regex is not None and self.speaker_regex.groups != 1:
            raise ValueError("speaker_regex must contain exactly one capture group")
        self.bootstrap_samples, self.seed = bootstrap_samples, seed

    def prepare(self):
        if not self.segments_path.is_file():
            raise FileNotFoundError(self.segments_path)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if any((self.out_dir / name).exists() for name in ("analysis.json", "analysis.csv", "analysis.md")):
            raise FileExistsError(f"Analysis already exists in {self.out_dir}; choose a new out_dir")

    def write_config(self, rendered_config):
        (self.out_dir / "config.yaml").write_text(rendered_config)

    def _rows(self):
        required = (
            set(SETTING_FIELDS)
            | {"utt_id", "duration_ms", "is_edge", "start_frame", "end_frame", "num_frames"}
            | set(RELATIONSHIPS.values())
        )
        with (
            self.segments_path.open() as handle,
            tqdm(
                total=self.segments_path.stat().st_size, unit="B", unit_scale=True, desc="Reading segments"
            ) as progress,
        ):
            for number, line in enumerate(handle, 1):
                progress.update(len(line.encode("utf-8")))
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    missing = required - row.keys()
                    if missing:
                        raise ValueError(f"Missing fields: {sorted(missing)}")
                    if not isinstance(row["is_edge"], bool):
                        raise ValueError("is_edge must be boolean")
                    for name in ("duration_ms", *RELATIONSHIPS.values()):
                        value = row[name]
                        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                            raise ValueError(f"Invalid finite measurement: {name}")
                    if row["duration_ms"] is None or row["duration_ms"] <= 0:
                        raise ValueError("duration_ms must be positive")
                    yield row
                except (ValueError, TypeError) as error:
                    raise ValueError(f"Invalid segment record at line {number}: {error}") from error

    def _interval(self, speakers, generator):
        # Within-utterance ranks are centered before accumulation: the sums
        # below can therefore be resampled across speakers without recentering.
        if not self.speaker_regex or len(speakers) < 2 or not self.bootstrap_samples:
            return None
        stats = torch.tensor([[s.cross, s.x.m2, s.y.m2] for s in speakers.values()], dtype=torch.float64)
        indices = torch.randint(len(stats), (self.bootstrap_samples, len(stats)), generator=generator)
        sums = stats[indices].sum(1)
        denominators = (sums[:, 1] * sums[:, 2]).sqrt()
        valid = denominators > 0
        if not valid.any():
            return None
        return torch.quantile(
            sums[valid, 0] / denominators[valid], torch.tensor([0.025, 0.975], dtype=torch.float64)
        ).tolist()

    def run(self):
        self.prepare()
        groups, utterances, row_count = {}, set(), 0

        # Group only consecutive rows: never materialize the complete JSONL.
        def key(row):
            return tuple(row[field] for field in SETTING_FIELDS), row["utt_id"]

        for (setting, utt), records in groupby(self._rows(), key=key):
            rows = list(records)
            row_count += len(rows)
            identity = (setting[0], utt)
            utterances.add(identity)
            if self.speaker_regex:
                match = self.speaker_regex.search(utt)
                if match is None:
                    raise ValueError(f"speaker_regex did not match utterance {utt!r}")
                speaker = match.group(1)
            else:
                speaker = None
            if setting not in groups:
                groups[setting] = dict(
                    utts=set(),
                    speakers={},
                    durations=RunningMoments(),
                    edges=0,
                    relationships={
                        name: dict(raw=_Pairs(), within=_Pairs(), speakers={}, utterances=0) for name in RELATIONSHIPS
                    },
                    bins={side: [RunningMoments() for _ in range(4)] for side in ("left_strength", "right_strength")},
                )
            group = groups[setting]
            if identity in group["utts"]:
                raise ValueError(f"Noncontiguous or duplicate segmentation for {utt!r}; use original pipeline output")
            group["utts"].add(identity)
            if (
                rows[0]["start_frame"] != 0
                or not rows[0]["is_edge"]
                or not rows[-1]["is_edge"]
                or rows[-1]["end_frame"] != rows[-1]["num_frames"]
            ):
                raise ValueError(f"Incomplete segmentation for {utt!r}")
            if any(a["end_frame"] != b["start_frame"] for a, b in zip(rows[:-1], rows[1:], strict=True)):
                raise ValueError(f"Noncontiguous frame coordinates for {utt!r}")
            interior = [r for r in rows if not r["is_edge"]]
            group["edges"] += len(rows) - len(interior)
            duration = torch.tensor([r["duration_ms"] for r in interior], dtype=torch.float64)
            group["durations"].update(duration)
            if speaker is not None:
                state = group["speakers"].setdefault(speaker, dict(utterances=0, duration=RunningMoments()))
                state["utterances"] += 1
                state["duration"].update(duration)
            for name, field in RELATIONSHIPS.items():
                # Neighbor pairs require both segments to be interior; all
                # other relations use the current interior segment.
                eligible = [
                    r
                    for i, r in enumerate(rows)
                    if not r["is_edge"]
                    and r[field] is not None
                    and (name != "next_duration" or (i + 1 < len(rows) and not rows[i + 1]["is_edge"]))
                ]
                x = torch.tensor([r[field] for r in eligible], dtype=torch.float64)
                y = torch.tensor([r["duration_ms"] for r in eligible], dtype=torch.float64)
                relation = group["relationships"][name]
                relation["raw"].update(x, y)
                if len(x) >= 3 and x.max() != x.min() and y.max() != y.min():
                    # Percentile ranks prevent long utterances receiving
                    # cubic weight from unnormalized rank magnitudes.
                    rx, ry = average_ranks(x) / len(x), average_ranks(y) / len(y)
                    rx, ry = rx - rx.mean(), ry - ry.mean()
                    relation["within"].update(rx, ry)
                    relation["utterances"] += 1
                    if speaker is not None:
                        relation["speakers"].setdefault(speaker, _Pairs()).update(rx, ry)
            for side in group["bins"]:
                eligible = [r for r in interior if r[side] is not None]
                if not eligible:
                    continue
                strength = torch.tensor([r[side] for r in eligible], dtype=torch.float64)
                durations = torch.tensor([r["duration_ms"] for r in eligible], dtype=torch.float64)
                # Average ties; a constant-strength utterance occupies one bin.
                ranks = average_ranks(strength) / len(strength)
                for index, moments in enumerate(group["bins"][side]):
                    moments.update(durations[(ranks > index / 4) & (ranks <= (index + 1) / 4)])
        if not row_count:
            raise ValueError("No segment records found")
        generator = torch.Generator().manual_seed(self.seed)

        def summary(moments):
            return {
                k: v if not isinstance(v, float) or math.isfinite(v) else None for k, v in moments.summary().items()
            }

        results = []
        for setting, group in groups.items():
            item = dict(zip(SETTING_FIELDS, setting, strict=True))
            item.update(
                utterances=len(group["utts"]),
                speakers=len(group["speakers"]) if self.speaker_regex else None,
                interior_duration_ms=summary(group["durations"]),
                excluded_edge_segments=group["edges"],
            )
            item["relationships"] = {
                name: dict(
                    pooled_pearson=state["raw"].correlation(),
                    pairs=state["raw"].x.count,
                    within_utterance_rank_correlation=state["within"].correlation(),
                    rank_pairs=state["within"].x.count,
                    rank_utterances=state["utterances"],
                    speaker_bootstrap_95_interval=self._interval(state["speakers"], generator),
                )
                for name, state in group["relationships"].items()
            }
            item["strength_quartiles"] = {
                side: [dict(quartile=i + 1, duration_ms=summary(m)) for i, m in enumerate(bins)]
                for side, bins in group["bins"].items()
            }
            item["by_speaker"] = {
                speaker: dict(utterances=state["utterances"], interior_duration_ms=summary(state["duration"]))
                for speaker, state in group["speakers"].items()
            }
            results.append(item)
        report = dict(
            version=1,
            source=str(self.segments_path),
            rows=row_count,
            unique_utterances=len(utterances),
            speaker_regex=self.speaker_regex.pattern if self.speaker_regex else None,
            seed=self.seed,
            bootstrap_samples=self.bootstrap_samples,
            methodology="Interior segments only. Pearson is pooled. Rank correlations pool centered within-utterance percentile ranks; constant variables and utterances with fewer than 3 pairs are excluded from ranks. Neighbor pairs require both segments interior. Intervals resample speakers, retaining all their utterances; unavailable without speaker IDs or fewer than 2 speakers. Strength quartiles use within-utterance average percentile ranks. These are detector-conditioned descriptive associations, not causal effects. No cross-layer comparisons are computed.",
            groups=results,
        )
        (self.out_dir / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        csv_rows = []
        for item in results:
            for name, relation in item["relationships"].items():
                csv_rows.append(
                    {
                        **{field: item[field] for field in SETTING_FIELDS},
                        "relationship": name,
                        **{k: v for k, v in relation.items() if k != "speaker_bootstrap_95_interval"},
                        "speaker_bootstrap_95_lower": relation["speaker_bootstrap_95_interval"][0]
                        if relation["speaker_bootstrap_95_interval"]
                        else None,
                        "speaker_bootstrap_95_upper": relation["speaker_bootstrap_95_interval"][1]
                        if relation["speaker_bootstrap_95_interval"]
                        else None,
                    }
                )
        with (self.out_dir / "analysis.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        lines = [
            "# Boundary segment analysis",
            "",
            f"Read {row_count:,} rows across {len(utterances):,} utterances.",
            "",
            report["methodology"],
            "",
            "| Layer axis index | Cutoff | Spacing (ms) | Relationship | Within-utterance rank r | Speaker bootstrap 95% interval |",
            "|---|---|---|---|---|---|",
        ]

        def fmt(value):
            return "n/a" if value is None else f"{value:.3f}"

        for item in results:
            for name, relation in item["relationships"].items():
                interval = relation["speaker_bootstrap_95_interval"]
                lines.append(
                    f"| {item['layer_index']} | {item['threshold']} | {item['min_distance_ms']} | {name} | {fmt(relation['within_utterance_rank_correlation'])} | {', '.join(map(fmt, interval)) if interval else 'n/a'} |"
                )
        (self.out_dir / "analysis.md").write_text("\n".join(lines) + "\n")
        print(f"Wrote analysis.json, analysis.csv and analysis.md to {self.out_dir}")
        return report
