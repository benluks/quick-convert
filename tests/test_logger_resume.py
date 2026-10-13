from functools import partial
from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("lightning")

from lightning.pytorch.loggers import WandbLogger

from quick_convert.training.lightning.logging.resume import LoggerRunCheckpoint, resolve_loggers


class RecordingWandbLogger(WandbLogger):
    def __init__(self, **options):
        self.options = options

    @property
    def version(self):
        return self.options.get("id")


def checkpoint(tmp_path, runs=None):
    path = tmp_path / "saved.ckpt"
    torch.save({"quick_convert_logger_runs": runs or []}, path)
    return path


def identity(run_id="tezj8b44"):
    return {"backend": "wandb", "id": run_id, "entity": "team", "project": "example_training"}


def test_metadata_restores_identity_before_logger_construction_and_preserves_factory(tmp_path):
    path = checkpoint(tmp_path, [identity()])
    factory = partial(RecordingWandbLogger, project="example_training", name="experiment", id=None, entity=None)
    logger = resolve_loggers(factory, path)
    assert logger.options == {
        "project": "example_training",
        "name": "experiment",
        "id": "tezj8b44",
        "entity": "team",
        "resume": "must",
    }
    assert factory.keywords["id"] is None


def test_legacy_layout_recovers_run_id_and_project(tmp_path):
    path = tmp_path / "example_training" / "tezj8b44" / "checkpoints" / "last.ckpt"
    path.parent.mkdir(parents=True)
    torch.save({}, path)
    logger = resolve_loggers(partial(RecordingWandbLogger, project="example_training", entity="team"), path)
    assert logger.options["id"] == "tezj8b44"
    assert logger.options["resume"] == "must"


@pytest.mark.parametrize("options", [{"id": "chosen"}, {"version": "chosen"}, {"resume": "never"}, {"resume": None}])
def test_explicit_overrides_do_not_require_reading_checkpoint(options):
    logger = resolve_loggers(partial(RecordingWandbLogger, **options), "/does/not/exist.ckpt")
    for key, value in options.items():
        assert logger.options[key] == value
    if "resume" not in options:
        assert logger.options["resume"] == "must"


def test_explicit_allow_mode_is_preserved_during_identity_recovery(tmp_path):
    logger = resolve_loggers(partial(RecordingWandbLogger, resume="allow"), checkpoint(tmp_path, [identity()]))
    assert logger.options["resume"] == "allow"
    assert logger.options["id"] == "tezj8b44"


def test_missing_or_mismatched_identity_fails_before_logger_initialization(tmp_path):
    with pytest.raises(ValueError, match="Cannot recover"):
        resolve_loggers(partial(RecordingWandbLogger, project="example_training"), checkpoint(tmp_path))
    path = checkpoint(tmp_path, [identity()])
    with pytest.raises(ValueError, match="project differs"):
        resolve_loggers(partial(RecordingWandbLogger, project="other"), path)
    with pytest.raises(ValueError, match="entity differs"):
        resolve_loggers(partial(RecordingWandbLogger, entity="other"), path)


def test_multiple_identities_require_an_unambiguous_match(tmp_path):
    second = {**identity("second"), "project": "second-project"}
    path = checkpoint(tmp_path, [identity(), second])
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_loggers(partial(RecordingWandbLogger), path)
    logger = resolve_loggers(partial(RecordingWandbLogger, project="second-project"), path)
    assert logger.options["id"] == "second"


def test_fresh_and_non_wandb_logging_remain_unchanged():
    factory = partial(RecordingWandbLogger, project="fresh")
    assert resolve_loggers(factory).options == {"project": "fresh"}
    marker = object()
    assert resolve_loggers(partial(lambda: marker), "/unused.ckpt") is marker
    for value in (None, True, False):
        assert resolve_loggers(value, "/unused.ckpt") is value
    assert resolve_loggers([marker, factory])[0] is marker
    with pytest.raises(ValueError, match="deferred"):
        resolve_loggers(RecordingWandbLogger(), "/unused.ckpt")


def test_callback_persists_identity_outside_model_state(monkeypatch):
    logger = RecordingWandbLogger()
    monkeypatch.setattr(
        RecordingWandbLogger,
        "experiment",
        property(lambda self: SimpleNamespace(id="tezj8b44", entity="team", project="example_training")),
    )
    state = {"state_dict": {"system.weight": torch.ones(1)}}
    LoggerRunCheckpoint().on_save_checkpoint(SimpleNamespace(loggers=[logger]), None, state)
    assert state["quick_convert_logger_runs"] == [identity()]
    assert list(state["state_dict"]) == ["system.weight"]


def test_hydra_logger_is_deferred_and_prepare_registers_callback(tmp_path, monkeypatch):
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from quick_convert.training.lightning import trainer as backend

    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).parents[1] / "quick_convert/configs")):
        config = compose(
            config_name="trainer/lightning",
            overrides=[
                "trainer.module=null",
                "trainer.trainer_kwargs.logger.project=example_training",
                "trainer.trainer_kwargs.logger.name=experiment",
                "trainer.trainer_kwargs.logger.save_dir=outputs",
            ],
        )
    assert isinstance(instantiate(config.trainer.trainer_kwargs.logger), partial)
    path = checkpoint(tmp_path, [identity()])
    loggers = []

    def create_trainer(**kwargs):
        loggers.append(kwargs["logger"])
        assert sum(isinstance(c, LoggerRunCheckpoint) for c in kwargs["callbacks"]) == 1
        return SimpleNamespace(loggers=[], default_root_dir=kwargs["default_root_dir"])

    monkeypatch.setattr(backend.L, "Trainer", create_trainer)
    trainer = backend.LightningTrainer(
        module=SimpleNamespace(setup_training=lambda _: None),
        ckpt_path=str(path),
        trainer_kwargs={"logger": partial(RecordingWandbLogger), "callbacks": [LoggerRunCheckpoint()]},
    )
    trainer.prepare(None, tmp_path)
    assert loggers[0].options["id"] == "tezj8b44"


def test_real_lightning_logger_receives_identity_without_starting_wandb_early(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    calls = []
    run = SimpleNamespace(id="tezj8b44", entity="team", project="example_training")

    def initialize(**options):
        calls.append(options)
        return run

    monkeypatch.setattr(wandb, "run", None)
    monkeypatch.setattr(wandb, "init", initialize)
    logger = resolve_loggers(
        partial(WandbLogger, project="example_training", save_dir=str(tmp_path)), checkpoint(tmp_path, [identity()])
    )
    assert not calls
    assert logger.experiment is run
    assert calls[0]["id"] == "tezj8b44"
    assert calls[0]["entity"] == "team"
    assert calls[0]["resume"] == "must"
    state = {}
    LoggerRunCheckpoint().on_save_checkpoint(SimpleNamespace(loggers=[logger]), None, state)
    assert state["quick_convert_logger_runs"] == [identity()]
