"""Small local HTML/JSON reports for real retrieved SSL trajectory pairs."""

from __future__ import annotations

import base64
import html
import io
import json
import math
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

from quick_convert.components.boundaries.inspection import compare_trajectories, segment_key, select_pairs
from quick_convert.data import AudioBatch


def _heatmap(matrix, title):
    rows, cols = len(matrix), len(matrix[0])
    cells = []
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            # Fixed absolute [0,2] cosine-distance scale shared by every panel.
            fraction = max(0.0, min(1.0, value / 2))
            rgb = tuple(round(a + (b - a) * fraction) for a, b in zip((239, 246, 255), (109, 40, 217), strict=True))
            cells.append(
                f'<rect x="{40 + j * 240 / cols:.3f}" y="{24 + i * 180 / rows:.3f}" width="{240 / cols + 0.02:.3f}" height="{180 / rows + 0.02:.3f}" fill="rgb{rgb}"><title>{value:.4f}</title></rect>'
            )
    return f'<svg role="img" aria-label="{html.escape(title, quote=True)}" viewBox="0 0 320 245"><text x="40" y="14">{html.escape(title)}</text>{"".join(cells)}<path d="M40 24 L280 204" stroke="#334155" stroke-width="1" stroke-dasharray="3 3" fill="none"/><text x="40" y="223">candidate: start → end</text><text transform="translate(14 200) rotate(-90)">query: start → end</text><text x="40" y="240">cosine distance: 0 pale → 2 purple</text></svg>'


def _profile(diagnostics, random_diagnostics):
    series = [("retrieved", diagnostics["pointwise_distances"], "#6d28d9")]
    series.extend(
        (f"self stretch {r['requested_factor']:g}×", r["pointwise_distances"], color)
        for r, color in zip(diagnostics["stretch_references"], ("#0f766e", "#2563eb", "#d97706"), strict=False)
    )
    if random_diagnostics is not None:
        series.append(("random partner", random_diagnostics["pointwise_distances"], "#b45309"))
    paths = []
    for index, (label, values, color) in enumerate(series):
        points = " ".join(f"{40 + i * 520 / (len(values) - 1):.3f},{175 - v * 75:.3f}" for i, v in enumerate(values))
        paths.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/><text x="{40 + (index % 3) * 190}" y="{210 + (index // 3) * 18}" fill="{color}">{html.escape(label)}</text>'
        )
    ticks = "".join(
        f'<text x="10" y="{179 - v * 75}">{v:g}</text><path d="M40 {175 - v * 75} H560" stroke="#e2e8f0"/>'
        for v in (0, 1, 2)
    )
    return f'<svg role="img" aria-label="Distance by normalized time" viewBox="0 0 640 250">{ticks}<text x="40" y="14">Pointwise cosine distance (absolute scale)</text>{"".join(paths)}<text x="40" y="193">0% relative time</text><text x="455" y="193">100%</text></svg>'


def _render_report(report, audio):
    esc = html.escape
    cards = []
    for index, pair in enumerate(report["pairs"]):
        record = pair["record"]
        q, c = record["query"], record["candidate"]
        d = pair["diagnostics"]
        ref = d["stretch_references"][-1]
        players = ""
        for label, segment in (("Query", q), ("Candidate", c)):
            source = audio.get(segment_key(segment))
            if source:
                players += f'<div><strong>{label} with context</strong><audio controls preload="none" src="data:audio/wav;base64,{source}"></audio></div>'
        random = pair["random_diagnostics"]
        random_text = (
            f"Random-partner distance: {random['cosine_distance']:.4f}."
            if random
            else "No saved random partner for this query."
        )
        cards.append(f"""<section id="pair-{index + 1}"><h2>Pair {index + 1} · score stratum {pair["score_stratum"]} · {esc(pair["duration_bucket"].replace("_", " "))}</h2>
<p><strong>{q["duration_ms"]:.1f} ms vs {c["duration_ms"]:.1f} ms ({record["duration_ratio"]:.2f}×)</strong> · saved distance {record["cosine_distance"]:.4f} · recomputed {d["cosine_distance"]:.4f} · mean-vector distance {d["mean_descriptor_cosine_distance"]:.4f}</p>
<p>Query: {esc(q["utt_id"])}, {q["start_seconds"]:.3f}–{q["end_seconds"]:.3f} s<br>Candidate: {esc(c["utt_id"])}, {c["start_seconds"]:.3f}–{c["end_seconds"]:.3f} s</p>
<div class="players">{players}</div>
<div class="grid">{_heatmap(d["native_distance_matrix"], "Native-frame comparison")}{_heatmap(d["normalized_distance_matrix"], f"Normalized real pair ({d['descriptor_points']} points)")}{_heatmap(ref["normalized_distance_matrix"], f"Known self stretch ({ref['requested_factor']:g}×)")}</div>
<p>Native grid: {d["query_native_frames"]} × {d["candidate_native_frames"]} frames; display grids may be capped. Dashed diagonals indicate equal relative time, not an inferred alignment.</p>
{_profile(d, random)}<p>{random_text} Self-stretch distance: {ref["cosine_distance"]:.6f}. Lower distances mean closer SSL vectors; no equivalence threshold is established.</p></section>""")
    notes = esc(report["interpretation"])
    baseline_rows = "".join(
        f"<tr><td>{esc(name.replace('_', ' '))}</td><td>{state['cosine_distance']['mean']:.4f}</td><td>{state['duration_ratio']['p50']:.2f}×</td></tr>"
        for name, state in report.get("run_level_conditions", {}).items()
        if state["cosine_distance"]["mean"] is not None and state["duration_ratio"]["p50"] is not None
    )
    baseline = (
        f"<h2>Original run summary (all sampled queries)</h2><table><tr><th>Condition</th><th>Mean cosine distance</th><th>Median duration ratio</th></tr>{baseline_rows}</table>"
        if baseline_rows
        else ""
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SSL trajectory pair inspection</title><style>
body{{font:16px system-ui,sans-serif;color:#172033;background:#f1f5f9;margin:0;padding:24px;line-height:1.5}}main{{max-width:1200px;margin:auto}}section,header{{background:white;padding:24px;margin-bottom:24px;border-radius:12px}}h1,h2{{line-height:1.2}}h2{{font-size:21px}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}svg{{width:100%;height:auto}}svg text{{font:10px system-ui,sans-serif}}section>svg{{max-width:800px}}.players{{display:flex;gap:24px;flex-wrap:wrap}}audio{{display:block;max-width:100%}}@media(max-width:750px){{.grid{{grid-template-columns:1fr}}body{{padding:10px}}}}
</style></head><body><main><header><h1>SSL trajectory pair inspection</h1><p>{len(report["pairs"])} selected pairs · {report["encoded_utterances"]} source utterances re-encoded · {report["descriptor_points"]} relative-time points</p><p>{notes}</p>{baseline}<p>Selection covers score-rank strata crossed with duration differences; it is deliberately not a representative sample. Audio includes {report["audio_context_seconds"]:g} seconds on each side when available.</p></header>{"".join(cards)}</main></body></html>"""


class InspectTrajectoriesPipeline:
    """Re-encode only source utterances needed by a small saved-match sample."""

    def __init__(
        self,
        input_dir,
        encoder,
        out_dir,
        population="detected",
        condition="nearest",
        score_bins=3,
        pairs_per_stratum=2,
        duration_ratio_split=1.5,
        stretch_factors=(1.5, 2.0),
        max_display_frames=128,
        include_audio=True,
        audio_context_seconds=0.15,
        path_prefix_from=None,
        path_prefix_to=None,
        score_tolerance=0.001,
    ):
        if (
            not isinstance(score_bins, int)
            or not isinstance(pairs_per_stratum, int)
            or score_bins < 1
            or pairs_per_stratum < 1
        ):
            raise ValueError("Inspection strata counts must be positive integers")
        if population not in ("detected", "random_location") or condition not in ("nearest", "dtw_refined"):
            raise ValueError("Choose a real, ordered retrieval population/condition")
        if not math.isfinite(duration_ratio_split) or duration_ratio_split <= 1:
            raise ValueError("duration_ratio_split must exceed 1")
        if (
            not stretch_factors
            or len(stretch_factors) > 3
            or any(not math.isfinite(x) or x <= 1 for x in stretch_factors)
        ):
            raise ValueError("Choose 1–3 finite stretch factors greater than 1")
        if not isinstance(max_display_frames, int) or max_display_frames < 2:
            raise ValueError("max_display_frames must be >=2")
        if (
            not math.isfinite(audio_context_seconds)
            or audio_context_seconds < 0
            or not math.isfinite(score_tolerance)
            or score_tolerance < 0
        ):
            raise ValueError("Invalid audio context or score tolerance")
        if (path_prefix_from is None) != (path_prefix_to is None):
            raise ValueError("Both path prefixes must be supplied together")
        self.input_dir, self.encoder, self.out_dir = Path(input_dir), encoder, Path(out_dir)
        self.population, self.condition, self.score_bins = population, condition, score_bins
        self.pairs_per_stratum, self.duration_ratio_split = pairs_per_stratum, duration_ratio_split
        self.stretch_factors, self.max_display_frames = tuple(stretch_factors), max_display_frames
        self.include_audio, self.audio_context_seconds = include_audio, audio_context_seconds
        self.path_prefix_from, self.path_prefix_to = path_prefix_from, path_prefix_to
        self.score_tolerance = score_tolerance

    def prepare(self):
        for name in ("trajectory_analysis.json", "trajectory_matches.jsonl"):
            if not (self.input_dir / name).is_file():
                raise FileNotFoundError(self.input_dir / name)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if any((self.out_dir / name).exists() for name in ("inspection.json", "inspection.html")):
            raise FileExistsError(f"Inspection already exists in {self.out_dir}; choose a new out_dir")

    def write_config(self, rendered_config):
        (self.out_dir / "config.yaml").write_text(rendered_config)

    @torch.inference_mode()
    def run(self):
        self.prepare()
        source = json.loads((self.input_dir / "trajectory_analysis.json").read_text())
        points = source["descriptor_points"]
        with (self.input_dir / "trajectory_matches.jsonl").open() as handle:
            selected = select_pairs(
                (json.loads(line) for line in handle if line.strip()),
                population=self.population,
                condition=self.condition,
                score_bins=self.score_bins,
                pairs_per_stratum=self.pairs_per_stratum,
                duration_ratio_split=self.duration_ratio_split,
            )
        requests = {}
        for pair in selected:
            segments = [pair["record"]["query"], pair["record"]["candidate"]]
            if pair["random_partner"] is not None:
                segments.append(pair["random_partner"]["candidate"])
            for segment in segments:
                requests[segment_key(segment)] = segment
        by_path = defaultdict(list)
        for key, segment in requests.items():
            by_path[segment["path"]].append((key, segment))
        self.encoder.eval()
        cached, audio = {}, {}
        for path, segments in tqdm(by_path.items(), desc="Re-encoding selected source utterances"):
            actual = Path(path)
            if self.path_prefix_from is not None:
                actual = Path(self.path_prefix_to) / actual.relative_to(self.path_prefix_from)
            batch = AudioBatch.from_paths(actual, target_sr=self.encoder.sample_rate, mono=True)
            content = self.encoder(batch)
            expected = source["encoder"]
            actual_identity = dict(
                model_name=content.model_name,
                backend=content.backend,
                encoder_layer=content.layer,
                frame_hz=content.frame_hz,
            )
            if actual_identity != expected:
                raise ValueError(f"Encoder metadata differs from original run: {actual_identity} != {expected}")
            if content.values.ndim != 3 or len(content.values) != 1 or content.representation_type != "continuous":
                raise ValueError("Expected one continuous selected-layer [1,T,D] representation")
            frames = int(content.lengths[0])
            values = content.values[0, :frames].detach().cpu().float()
            for key, segment in segments:
                start, end = segment["start_frame"], segment["end_frame"]
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or start < 0
                    or end > frames
                    or end - start < 2
                ):
                    raise ValueError(f"Invalid saved segment bounds for {path}: {start}:{end}, valid frames={frames}")
                cached[key] = values[start:end].clone()
                if self.include_audio:
                    sr = int(batch.sample_rates[0])
                    a = max(0, int((segment["start_seconds"] - self.audio_context_seconds) * sr))
                    b = min(
                        int(batch.lengths[0]), math.ceil((segment["end_seconds"] + self.audio_context_seconds) * sr)
                    )
                    buffer = io.BytesIO()
                    sf.write(buffer, batch.waveforms[0, a:b].numpy(), sr, format="WAV", subtype="PCM_16")
                    audio[key] = base64.b64encode(buffer.getvalue()).decode("ascii")
            del values, content, batch
        for pair in selected:
            record = pair["record"]
            query = cached[segment_key(record["query"])]
            candidate = cached[segment_key(record["candidate"])]
            diagnostics = compare_trajectories(
                query,
                candidate,
                points=points,
                max_display_frames=self.max_display_frames,
                stretch_factors=self.stretch_factors,
            )
            drift = abs(diagnostics["cosine_distance"] - record["cosine_distance"])
            if drift > self.score_tolerance:
                raise ValueError(
                    f"Recomputed score differs by {drift:.6f} for {record['query']['utt_id']}; check source audio, encoder settings and precision"
                )
            pair["diagnostics"] = diagnostics
            pair["saved_score_absolute_difference"] = drift
            pair["random_diagnostics"] = (
                compare_trajectories(
                    query,
                    cached[segment_key(pair["random_partner"]["candidate"])],
                    points=points,
                    max_display_frames=self.max_display_frames,
                    stretch_factors=self.stretch_factors,
                )
                if pair["random_partner"]
                else None
            )
        report = dict(
            version=1,
            source_dir=str(self.input_dir),
            encoder=source["encoder"],
            descriptor_points=points,
            encoded_utterances=len(by_path),
            selected_segments=len(requests),
            population=self.population,
            condition=self.condition,
            run_level_conditions=source.get("populations", {}).get(self.population, {}).get("conditions", {}),
            selection=dict(
                score_bins=self.score_bins,
                pairs_per_stratum=self.pairs_per_stratum,
                duration_ratio_split=self.duration_ratio_split,
            ),
            audio_context_seconds=self.audio_context_seconds,
            include_audio=self.include_audio,
            interpretation="Whole source utterances are re-encoded before slicing saved segments, preserving SSL context. Heatmaps and curves use cosine distance on a shared absolute scale from 0 to 2. A pale diagonal band indicates lower-distance pointwise correspondence, not proven content identity. Stretched-copy references interpolate already-extracted SSL features, not waveforms; their low distances validate numerical invariance only and do not provide an acceptance threshold for natural pairs. Comparisons preserve SSL location; mean-vector distance is a separate diagnostic for static proximity. Native heatmaps may be display-subsampled; normalized scores use all original resampling points. No external acoustic or linguistic alignment is used.",
            pairs=selected,
        )
        # Render only after every pair passes reproducibility checks.
        (self.out_dir / "inspection.html").write_text(_render_report(report, audio))
        (self.out_dir / "inspection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Wrote inspection.html and inspection.json to {self.out_dir}")
        return report
