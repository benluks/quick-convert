from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from quick_convert.cli.run import execute_pipeline
from quick_convert.pipelines.training.pipeline import TrainingPipeline
from quick_convert.training.base import BaseTrainer


class RecordingBackend(BaseTrainer):
    def __init__(self, out_dir):
        super().__init__(out_dir)
        self.events = []

    def log_config(self, config, config_path):
        assert config_path.exists()
        assert OmegaConf.to_container(OmegaConf.load(config_path)) == config
        self.events.append(("config", config))

    def train(self, train_dataset=None, val_dataset=None):
        self.events.append(("train",))
        return "trained"


def test_runner_logs_resolved_config_after_file_before_training(monkeypatch, tmp_path):
    backend = RecordingBackend(tmp_path)
    pipeline = TrainingPipeline(backend, train_dataset=object())
    cfg = OmegaConf.create({"batch_size": 2, "trainer": {"batch_size": "${batch_size}"}, "pipeline": {}})
    monkeypatch.setattr("quick_convert.cli.run.instantiate", lambda config: pipeline)
    assert execute_pipeline(cfg) == "trained"
    assert backend.events == [("config", {"batch_size": 2, "trainer": {"batch_size": 2}, "pipeline": {}}), ("train",)]


def test_pipeline_requires_prepare_before_logging(tmp_path):
    with pytest.raises(RuntimeError, match="Prepare"):
        TrainingPipeline(RecordingBackend(tmp_path), object()).log_config({})


def test_backend_without_logging_keeps_noop_contract(tmp_path):
    class PlainBackend(BaseTrainer):
        def train(self, train_dataset=None, val_dataset=None):
            return "trained"

    backend = PlainBackend(tmp_path)
    assert backend.log_config({"x": 1}, tmp_path / "config.yaml") is None


@pytest.mark.parametrize("global_zero", [True, False])
def test_lightning_logs_only_wandb_and_only_on_rank_zero(monkeypatch, tmp_path, global_zero):
    pytest.importorskip("lightning")
    from quick_convert.training.lightning.trainer import LightningTrainer

    calls = []

    class FakeWandbLogger:
        def log_hyperparams(self, config):
            calls.append(("params", config))

        @property
        def experiment(self):
            return SimpleNamespace(save=lambda *args, **kwargs: calls.append(("save", args, kwargs)))

    monkeypatch.setattr("lightning.pytorch.loggers.WandbLogger", FakeWandbLogger)
    trainer = LightningTrainer.__new__(LightningTrainer)
    trainer.pl_trainer = SimpleNamespace(is_global_zero=global_zero, loggers=[object(), FakeWandbLogger()])
    path = tmp_path / "config.yaml"
    path.write_text("batch_size: 2\n")
    trainer.log_config({"batch_size": 2}, path)
    if global_zero:
        assert calls == [
            ("params", {"batch_size": 2}),
            ("save", (str(path.resolve()),), {"base_path": str(tmp_path.resolve()), "policy": "now"}),
        ]
    else:
        assert calls == []
    trainer.pl_trainer.loggers = []
    trainer.log_config({}, path)
