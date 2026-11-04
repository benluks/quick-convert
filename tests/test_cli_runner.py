from types import SimpleNamespace

import pytest

from quick_convert.cli import run as run_module


class RecordingPipeline:
    def __init__(self):
        self.config = None
        self.run_kwargs = None
        self.prepared = False

    def prepare(self):
        self.prepared = True

    def write_config(self, config):
        assert self.prepared
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
    assert pipeline.prepared
    assert pipeline.config == "rendered: true\n"
    assert pipeline.run_kwargs == {"flag": True}


def test_execute_pipeline_rejects_non_mapping_run_config(monkeypatch) -> None:
    cfg = SimpleNamespace(pipeline=object(), get=lambda key, default: ["invalid"] if key == "run" else default)
    monkeypatch.setattr(run_module.OmegaConf, "to_yaml", lambda cfg, resolve: "config")
    monkeypatch.setattr(run_module, "instantiate", lambda value: RecordingPipeline())

    with pytest.raises(TypeError, match="must be a mapping"):
        run_module.execute_pipeline(cfg)


def test_seed_is_applied_before_pipeline_instantiation(monkeypatch):
    import random

    import numpy as np
    import torch
    from omegaconf import OmegaConf

    samples = []

    def instantiate(config):
        samples.append((random.random(), np.random.random(), torch.rand(1).item()))
        return RecordingPipeline()

    monkeypatch.setattr(run_module, "instantiate", instantiate)
    cfg = OmegaConf.create({"seed": 17, "pipeline": {}})
    run_module.execute_pipeline(cfg, print_config=False)
    run_module.execute_pipeline(cfg, print_config=False)
    assert samples[0] == samples[1]


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5, "17"])
def test_invalid_seed_is_rejected_before_instantiation(monkeypatch, seed):
    from omegaconf import OmegaConf

    monkeypatch.setattr(run_module, "instantiate", lambda config: pytest.fail("Invalid seed constructed a pipeline"))
    with pytest.raises(ValueError, match="seed must be an integer"):
        run_module.execute_pipeline(OmegaConf.create({"seed": seed, "pipeline": {}}), print_config=False)
