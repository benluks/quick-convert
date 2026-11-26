import json
from types import SimpleNamespace

from omegaconf import OmegaConf

from quick_convert.pipelines.benchmarking import batching


def test_bucketed_benchmark_serialization(monkeypatch, tmp_path):
    monkeypatch.setattr(batching, "resolve_durations", lambda *args, **kwargs: [1.0, 2.0, 3.0])
    strategies = OmegaConf.create({"bucketed": {"kind": "bucketed", "batch_size": 4, "boundaries": [1, 2, 3]}})
    output = tmp_path / "report.json"
    pipeline = batching.BatchingBenchmarkPipeline(SimpleNamespace(max_length=None, rows=[]), strategies, output=output)
    pipeline.run()
    report = json.loads(output.read_text())
    assert report["results"][0]["strategy_options"]["boundaries"] == [1, 2, 3]
