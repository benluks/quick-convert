import json
from types import SimpleNamespace

from quick_convert.pipelines.benchmarking import batching


def test_batching_benchmark(tmp_path, monkeypatch):
    lengths = list(range(1, 33))
    dataset = SimpleNamespace(rows=list(range(32)), max_length=None)
    calls = []

    def resolve(rows, *, cache_path):
        calls.append(rows)
        return lengths

    monkeypatch.setattr(batching, "resolve_durations", resolve)
    output = tmp_path / "report.json"
    pipeline = batching.BatchingBenchmarkPipeline(
        dataset,
        {
            "random": {"kind": "random", "batch_size": 4},
            "sorted": {"kind": "length_aware", "batch_size": 4, "pool_size": 32},
            "dynamic": {"kind": "length_aware", "batch_size": 8, "max_batch_cost": 64},
        },
        seeds=[0, 1],
        output=output,
    )
    report = pipeline.run()
    assert calls == [dataset.rows]
    assert json.loads(output.read_text()) == report
    assert report["intrinsic_attention_cost"] == sum(x**2 for x in lengths)
    assert len(report["results"]) == 6
    for random, ordered, dynamic in (report["results"][:3], report["results"][3:]):
        assert random["examples"] == ordered["examples"] == dynamic["examples"] == 32
        assert ordered["padded_volume"] < random["padded_volume"]
        assert dynamic["max_padded_batch_seconds"] <= 64
        for row in (random, ordered, dynamic):
            assert row["attention_cost"] >= report["intrinsic_attention_cost"]
            assert row["batch_linear_cost"]["p100"] == row["max_padded_batch_seconds"]
            assert 0 <= row["attention_padding_fraction"] < 1


def test_cost_percentiles():
    assert batching.cost_percentiles(list(range(1, 101))) == {
        "p50": 50,
        "p90": 90,
        "p95": 95,
        "p99": 99,
        "p100": 100,
    }
