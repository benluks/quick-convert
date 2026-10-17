import json

import numpy as np
import pandas as pd
import pytest

from quick_convert.pipelines.summarize_boundaries import SummarizeBoundariesPipeline


SETTING = dict(
    split="test",
    model_name="synthetic",
    backend="test",
    encoder_layer=None,
    frame_hz=50.0,
    layer_index=5,
    metric="cosine",
    smoothing_frames=1,
    threshold_mode="utterance_quantile",
    threshold=0.75,
    min_distance_ms=20,
    min_distance_frames=1,
)


def records():
    result = []
    for utt, strengths, durations in [
        ("a-1", [1, 2, 2, 4], [40, 80, 120, 160]),
        ("b-1", [4, 3, 2, 1], [160, 80, 120, 40]),
        ("b-2", [1, 3, 4, 2], [40, 160, 120, 80]),
    ]:
        values = [20, *durations, 20]
        start = 0
        for index, duration in enumerate(values):
            result.append(
                dict(
                    **SETTING,
                    utt_id=utt,
                    start_frame=start,
                    end_frame=start + duration // 20,
                    num_frames=sum(values) // 20,
                    duration_ms=duration,
                    is_edge=index in (0, len(values) - 1),
                    left_strength=strengths[index - 1] if 0 < index < len(values) - 1 else None,
                    right_strength=strengths[index - 1] if 0 < index < len(values) - 1 else None,
                    mean_internal_l2=1 / duration,
                    path_length_l2=(duration // 20 - 1) / duration,
                    net_displacement_l2=0.8 * (duration // 20 - 1) / duration,
                    next_duration_ms=values[index + 1] if index + 1 < len(values) else None,
                )
            )
            start += duration // 20
    return result


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_streaming_correlations_match_offline_reference(tmp_path):
    rows = records()
    source = tmp_path / "segments.jsonl"
    write_rows(source, rows)
    report = SummarizeBoundariesPipeline(
        source, tmp_path / "analysis", speaker_regex=r"^([^-]+)-", bootstrap_samples=100
    ).run()
    g = report["groups"][0]
    assert report["rows"] == 18 and report["unique_utterances"] == 3
    assert g["speakers"] == 2 and g["interior_duration_ms"]["count"] == 12
    df = pd.DataFrame([row for row in rows if not row["is_edge"]])
    x = df.left_strength.to_numpy()
    y = df.duration_ms.to_numpy()
    expected = float(np.corrcoef(x, y)[0, 1])
    assert g["relationships"]["preceding_strength"]["pooled_pearson"] == pytest.approx(expected)
    rank = df.groupby("utt_id")[["left_strength", "duration_ms"]].rank(pct=True)
    centered = rank - rank.groupby(df.utt_id).transform("mean")
    expected = float(np.corrcoef(centered.left_strength, centered.duration_ms)[0, 1])
    relation = g["relationships"]["preceding_strength"]
    assert relation["within_utterance_rank_correlation"] == pytest.approx(expected)
    assert relation["rank_pairs"] == 12 and relation["rank_utterances"] == 3
    assert len(relation["speaker_bootstrap_95_interval"]) == 2
    assert g["relationships"]["next_duration"]["pairs"] == 9
    assert g["by_speaker"]["b"]["utterances"] == 2
    assert sum(bin["duration_ms"]["count"] for bin in g["strength_quartiles"]["left_strength"]) == 12
    assert len(pd.read_csv(tmp_path / "analysis" / "analysis.csv")) == 6
    assert (tmp_path / "analysis" / "analysis.md").is_file()
    with pytest.raises(FileExistsError):
        SummarizeBoundariesPipeline(source, tmp_path / "analysis").run()


def test_settings_are_isolated_and_unknown_speakers_have_no_interval(tmp_path):
    rows = records()
    other = [dict(row, min_distance_ms=80, min_distance_frames=4) for row in rows]
    source = tmp_path / "segments.jsonl"
    write_rows(source, rows + other)
    report = SummarizeBoundariesPipeline(source, tmp_path / "analysis").run()
    assert len(report["groups"]) == 2
    assert report["unique_utterances"] == 3
    for group in report["groups"]:
        assert group["interior_duration_ms"]["count"] == 12
        assert group["speakers"] is None
        assert all(r["speaker_bootstrap_95_interval"] is None for r in group["relationships"].values())


def test_constant_variables_and_empty_interior_are_json_null(tmp_path):
    rows = records()[:6]
    for row in rows:
        if not row["is_edge"]:
            row["left_strength"] = 1.0
    source = tmp_path / "segments.jsonl"
    write_rows(source, rows)
    report = SummarizeBoundariesPipeline(source, tmp_path / "analysis").run()
    relation = report["groups"][0]["relationships"]["preceding_strength"]
    assert relation["pooled_pearson"] is None and relation["within_utterance_rank_correlation"] is None
    # A single all-edge segment is a valid no-boundaries utterance.
    source2 = tmp_path / "empty.jsonl"
    write_rows(source2, [dict(rows[0], start_frame=0, end_frame=1, num_frames=1, next_duration_ms=None)])
    report = SummarizeBoundariesPipeline(source2, tmp_path / "empty-analysis").run()
    assert report["groups"][0]["interior_duration_ms"]["count"] == 0
    assert report["groups"][0]["interior_duration_ms"]["mean"] is None


@pytest.mark.parametrize("problem", ["duplicate", "truncated", "invalid", "empty"])
def test_rejects_invalid_or_noncontiguous_inputs(tmp_path, problem):
    rows = records()
    if problem == "duplicate":
        rows += rows[:6]
    elif problem == "truncated":
        rows = rows[:-1]
    elif problem == "invalid":
        rows[1]["duration_ms"] = float("nan")
    else:
        rows = []
    source = tmp_path / "segments.jsonl"
    write_rows(source, rows)
    with pytest.raises(ValueError):
        SummarizeBoundariesPipeline(source, tmp_path / "analysis").run()
    assert not (tmp_path / "analysis" / "analysis.json").exists()


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_movement_budget_recovers_known_scaling_and_duration_bins(tmp_path, beta):
    rows = records()
    for row in rows:
        transitions = row["end_frame"] - row["start_frame"] - 1
        # Distinct utterance scales must not bias the within-utterance slope.
        scale = {"a-1": 3.0, "b-1": 20.0, "b-2": 100.0}[row["utt_id"]]
        path = scale * transitions**beta if transitions else 0.0
        row["path_length_l2"] = path
        row["net_displacement_l2"] = path * 0.6
        row["mean_internal_l2"] = path / transitions if transitions else 0.0
    source = tmp_path / "segments.jsonl"
    write_rows(source, rows)
    report = SummarizeBoundariesPipeline(
        source, tmp_path / "analysis", speaker_regex=r"^([^-]+)-", bootstrap_samples=200
    ).run()
    assert report["version"] == 2
    budget = report["groups"][0]["movement_budget"]
    for name in ("path_length_l2", "net_displacement_l2"):
        scaling = budget["log_scaling"][name]
        assert scaling["within_utterance_log_slope"] == pytest.approx(beta, abs=1e-12)
        assert scaling["speaker_bootstrap_95_interval"] == pytest.approx([beta, beta], abs=1e-12)
        assert scaling["pairs"] == 12
    interior = [row for row in rows if not row["is_edge"]]
    for bin in budget["duration_bins"]:
        selected = [
            row
            for row in interior
            if row["duration_ms"] >= bin["lower_ms"]
            and (bin["upper_ms"] is None or row["duration_ms"] < bin["upper_ms"])
        ]
        for name, summary in bin["measures"].items():
            assert summary["count"] == len(selected)
            if selected:
                assert summary["mean"] == pytest.approx(np.mean([row[name] for row in selected]))
    assert sum(bin["measures"]["duration_ms"]["count"] for bin in budget["within_utterance_duration_quartiles"]) == 12
    assert (tmp_path / "analysis" / "movement_budget.csv").is_file()


def test_zero_movement_and_one_frame_segments_are_not_log_observations(tmp_path):
    rows = records()
    for row in rows:
        row["path_length_l2"] = 0.0
        row["net_displacement_l2"] = 0.0
        row["mean_internal_l2"] = 0.0
    source = tmp_path / "segments.jsonl"
    write_rows(source, rows)
    report = SummarizeBoundariesPipeline(source, tmp_path / "analysis").run()
    budget = report["groups"][0]["movement_budget"]
    assert budget["measures"]["path_length_l2"]["count"] == 12
    assert budget["measures"]["path_length_l2"]["mean"] == 0.0
    assert budget["log_scaling"]["path_length_l2"]["within_utterance_log_slope"] is None
    assert budget["log_scaling"]["path_length_l2"]["pairs"] == 0


def test_one_frame_interiors_remain_in_bins_but_not_log_scaling(tmp_path):
    template = records()[0]
    rows = [
        dict(
            template,
            start_frame=i,
            end_frame=i + 1,
            num_frames=5,
            duration_ms=20.0,
            is_edge=i in (0, 4),
            left_strength=1.0 if i else None,
            right_strength=1.0 if i < 4 else None,
            next_duration_ms=20.0 if i < 4 else None,
            path_length_l2=0.0,
            net_displacement_l2=0.0,
            mean_internal_l2=0.0,
        )
        for i in range(5)
    ]
    source = tmp_path / "segments.jsonl"
    write_rows(source, rows)
    report = SummarizeBoundariesPipeline(source, tmp_path / "analysis").run()
    budget = report["groups"][0]["movement_budget"]
    assert budget["one_frame_segments"] == 3
    assert budget["duration_bins"][0]["measures"]["path_length_l2"]["count"] == 3
    assert budget["log_scaling"]["path_length_l2"]["eligible_positive_pairs"] == 0
