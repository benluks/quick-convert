from types import SimpleNamespace

import pytest

from quick_convert.cli import run as run_module


class RecordingPipeline:
    def __init__(self):
        self.config = None
        self.run_kwargs = None

    def write_config(self, config):
        self.config = config

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        return "result"


def test_execute_pipeline_handles_optional_capabilities(monkeypatch) -> None:
    pipeline = RecordingPipeline()
    cfg = SimpleNamespace(
        pipeline=object(),
        get=lambda key, default: {"flag": True} if key == "run" else default,
    )
    monkeypatch.setattr(run_module.OmegaConf, "to_yaml", lambda cfg, resolve: "rendered: true\n")
    monkeypatch.setattr(run_module, "instantiate", lambda value: pipeline)

    result = run_module.execute_pipeline(cfg)

    assert result == "result"
    assert pipeline.config == "rendered: true\n"
    assert pipeline.run_kwargs == {"flag": True}


def test_execute_pipeline_rejects_non_mapping_run_config(monkeypatch) -> None:
    cfg = SimpleNamespace(pipeline=object(), get=lambda key, default: ["invalid"])
    monkeypatch.setattr(run_module.OmegaConf, "to_yaml", lambda cfg, resolve: "config")
    monkeypatch.setattr(run_module, "instantiate", lambda value: RecordingPipeline())

    with pytest.raises(TypeError, match="must be a mapping"):
        run_module.execute_pipeline(cfg)
