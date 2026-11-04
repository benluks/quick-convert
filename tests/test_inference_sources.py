import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from quick_convert.inference import (
    export_inference_artifact,
    load_inference_artifact,
    load_inference_system,
    load_system_from_run,
    save_inference_artifact,
)
from quick_convert.inference.source import (
    CHECKPOINT_METADATA_KEY,
    InferenceState,
    inference_description,
    inference_system_config,
    restore_system,
)
from quick_convert.systems.base import InferenceSystem


class NewTask(InferenceSystem):
    def __init__(self, dimension=2, device="cpu"):
        super().__init__()
        self.projection = nn.Linear(dimension, dimension, device=device)

    def forward(self, values):
        return self.projection(values)


def task_config():
    return {"_target_": "test_inference_sources.NewTask", "dimension": 2}


def write_run(tmp_path, *, embedded=False):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    model = NewTask()
    config = {"system": task_config(), "training_only": {"_target_": "does.not.exist.Trainer"}}
    OmegaConf.save(OmegaConf.create(config), run / "config.yaml")
    saved = {
        "state_dict": {"system." + key: value for key, value in model.state_dict().items()},
        "optimizer_states": [{"training_only": True}],
    }
    saved["state_dict"]["training_only"] = torch.ones(1)
    if embedded:
        saved[CHECKPOINT_METADATA_KEY] = inference_description(config)
    torch.save(saved, run / "checkpoints/last.ckpt")
    return run, model


@pytest.mark.parametrize("embedded", [False, True])
def test_run_checkpoint_and_artifact_restore_identical_inference_only_models(tmp_path, embedded):
    run, original = write_run(tmp_path, embedded=embedded)
    before = sorted(path.relative_to(run) for path in run.rglob("*"))
    artifact = export_inference_artifact(run, tmp_path / "artifact")
    values = torch.ones(1, 2)
    for loaded in (
        load_system_from_run(run),
        load_inference_system(run),
        load_inference_system(run / "checkpoints/last.ckpt"),
        load_inference_system(artifact),
        load_inference_artifact(artifact),
        NewTask.from_run(run),
        NewTask.from_artifact(artifact),
    ):
        torch.testing.assert_close(loaded(values), original(values))
        assert not loaded.training
        assert all(not parameter.requires_grad for parameter in loaded.parameters())
        assert not hasattr(loaded, "training_only")
    assert sorted(path.relative_to(run) for path in run.rglob("*")) == before
    assert set(torch.load(artifact / "weights.pt", weights_only=True)) == set(original.state_dict())
    assert set(OmegaConf.load(artifact / "artifact.yaml")) == {
        "format",
        "version",
        "weights",
        "excluded_state_prefixes",
        "system",
    }


def test_embedded_checkpoint_works_after_removing_the_run_config(tmp_path):
    run, original = write_run(tmp_path, embedded=True)
    (run / "config.yaml").unlink()
    values = torch.ones(1, 2)
    artifact = export_inference_artifact(run, tmp_path / "artifact")
    for loaded in (
        load_inference_system(run / "checkpoints/last.ckpt"),
        NewTask.from_run(run),
        load_inference_system(artifact),
    ):
        torch.testing.assert_close(loaded(values), original(values))


def test_old_checkpoint_discovery_is_bounded_and_adjacent_config_wins(tmp_path):
    run, original = write_run(tmp_path)
    path = run / "checkpoints/last.ckpt"
    assert inference_system_config(path) == task_config()
    adjacent = path.parent / "config.yaml"
    adjacent.write_text("system:\n  _target_: torch.nn.Linear\n  in_features: 2\n  out_features: 2\n")
    assert inference_system_config(path)["_target_"] == "torch.nn.Linear"
    loaded = load_inference_system(path, config=(run / "config.yaml").resolve())
    torch.testing.assert_close(loaded(torch.ones(1, 2)), original(torch.ones(1, 2)))
    isolated = run / "unrelated/nested/standalone.ckpt"
    isolated.parent.mkdir(parents=True)
    torch.save(torch.load(path, weights_only=True), isolated)
    with pytest.raises(ValueError, match="no inference composition or nearby"):
        load_inference_system(isolated)


def test_explicit_config_overrides_embedded_metadata_and_selects_checkpoint(tmp_path):
    run, _ = write_run(tmp_path, embedded=True)
    (run / "alternative.yaml").write_text("system:\n  _target_: torch.nn.Linear\n  in_features: 2\n  out_features: 2\n")
    checkpoint = {
        "state_dict": {"system.weight": torch.full((2, 2), 7.0), "system.bias": torch.ones(2)},
        CHECKPOINT_METADATA_KEY: inference_description({"system": task_config()}),
    }
    path = run / "checkpoints/selected.ckpt"
    torch.save(checkpoint, path)
    loaded = load_system_from_run(run, checkpoint="checkpoints/selected.ckpt", config="alternative.yaml")
    assert isinstance(loaded, nn.Linear)
    torch.testing.assert_close(loaded.weight, torch.full((2, 2), 7.0))
    artifact = export_inference_artifact(run, tmp_path / "artifact", checkpoint=path, config="alternative.yaml")
    torch.testing.assert_close(load_inference_system(artifact).weight, loaded.weight)


def test_loader_rejects_ignored_options_and_unsupported_metadata(tmp_path):
    run, _ = write_run(tmp_path, embedded=True)
    artifact = export_inference_artifact(run, tmp_path / "artifact")
    for options in ({"checkpoint": "other.ckpt"}, {"config": "other.yaml"}):
        with pytest.raises(ValueError, match="omit checkpoint/config"):
            load_inference_system(artifact, **options)
    path = run / "checkpoints/last.ckpt"
    with pytest.raises(ValueError, match="already names a checkpoint"):
        load_inference_system(path, checkpoint="other.ckpt")
    checkpoint = torch.load(path, weights_only=True)
    checkpoint[CHECKPOINT_METADATA_KEY]["version"] = 999
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="format or version"):
        load_inference_system(path)
    checkpoint[CHECKPOINT_METADATA_KEY]["version"] = 1
    checkpoint[CHECKPOINT_METADATA_KEY]["adapter"] = "unavailable"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="unsupported inference adapter"):
        load_inference_system(path)


def test_restore_overrides_device_preserves_inputs_and_checks_weights(tmp_path):
    model = NewTask()
    config = {**task_config(), "device": "cuda"}
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    state = InferenceState(config, weights)
    loaded = restore_system(state, map_location="cpu")
    assert next(loaded.parameters()).device.type == "cpu"
    assert config["device"] == "cuda"
    assert all(value.requires_grad for value in model.parameters())
    for key, value in weights.items():
        torch.testing.assert_close(value, model.state_dict()[key])
    state.state_dict = {"projection.weight": weights["projection.weight"]}
    with pytest.raises(RuntimeError, match="Missing key"):
        restore_system(state)
    assert isinstance(restore_system(state, strict=False), NewTask)
    state.excluded_state_prefixes = ("projection.bias",)
    assert isinstance(restore_system(state), NewTask)
    assert "projection.bias" not in state.state_dict


def test_empty_parameter_system_and_compiled_keys_are_supported(tmp_path):
    run, model = write_run(tmp_path)
    saved = {"state_dict": {"system._orig_mod." + key: value for key, value in model.state_dict().items()}}
    torch.save(saved, run / "checkpoints/last.ckpt")
    torch.testing.assert_close(load_inference_system(run)(torch.ones(1, 2)), model(torch.ones(1, 2)))
    artifact = save_inference_artifact(nn.Identity(), {"_target_": "torch.nn.Identity"}, tmp_path / "identity")
    assert isinstance(load_inference_system(artifact), nn.Identity)
    OmegaConf.save(OmegaConf.create({"system": {"_target_": "torch.nn.Identity"}}), run / "config.yaml")
    torch.save({"state_dict": {}}, run / "checkpoints/last.ckpt")
    assert isinstance(load_inference_system(run), nn.Identity)


def test_resolves_only_system_and_checks_the_instantiated_type():
    config = OmegaConf.create(
        {"dimension": 2, "system": {**task_config(), "dimension": "${dimension}"}, "training": "???"}
    )
    assert inference_description(config)["system"] == task_config()
    assert inference_description({"pipeline": {"_target_": "training.only"}}) is None
    with pytest.raises(TypeError, match="torch.nn.Module"):
        restore_system(InferenceState({"_target_": "builtins.dict"}, {}))


def test_task_loaders_check_the_requested_class(tmp_path):
    class OtherTask(InferenceSystem):
        pass

    run, model = write_run(tmp_path)
    with pytest.raises(TypeError, match="expected OtherTask"):
        OtherTask.from_run(run)
    artifact = save_inference_artifact(model, task_config(), tmp_path / "artifact")
    with pytest.raises(TypeError, match="expected OtherTask"):
        OtherTask.from_artifact(artifact)


def assert_loads_without_optional_imports(tmp_path, *sources):
    code = """
import importlib.abc, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.startswith(('lightning', 'wandb', 'transformers', 'quick_convert.training')):
            raise AssertionError('Inference imported ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from quick_convert.inference import load_inference_system
for source in sys.argv[1:]:
    model = load_inference_system(source)
    assert not model.training
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(Path(__file__).parent), str(Path(__file__).parents[1])]))
    result = subprocess.run(
        [sys.executable, "-c", code, *map(str, sources)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_loading_needs_no_training_or_model_backend_imports(tmp_path):
    run, _ = write_run(tmp_path, embedded=True)
    artifact = export_inference_artifact(run, tmp_path / "artifact")
    assert_loads_without_optional_imports(tmp_path, run, run / "checkpoints/last.ckpt", artifact)


def test_reconstruction_checkpoint_does_not_pickle_optimization_objects(tmp_path):
    pytest.importorskip("lightning")
    from quick_convert.training.lightning.modules.ssl_reconstruction import SSLReconstructionTrainingModule
    from quick_convert.training.lightning.optim import Optimization

    module = SSLReconstructionTrainingModule(optimization=Optimization(lr_scheduler=None), system=NewTask())
    assert "optimization" not in module.hparams
    assert not module._log_hyperparams
    module.set_inference_config({"system": task_config()})
    checkpoint = {"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams)}
    module.on_save_checkpoint(checkpoint)
    path = tmp_path / "reconstruction.ckpt"
    torch.save(checkpoint, path)
    assert_loads_without_optional_imports(tmp_path, path)


def test_dependency_planner_inspects_the_saved_system_without_instantiation(tmp_path):
    from quick_convert.dependencies import dependency_plan
    from quick_convert.utils.device import override_devices

    run, _ = write_run(tmp_path, embedded=True)
    saved = torch.load(run / "checkpoints/last.ckpt", weights_only=True)
    saved[CHECKPOINT_METADATA_KEY]["system"] = {"_target_": "quick_convert.components.ssl.W2VBertContentEncoder"}
    torch.save(saved, run / "checkpoints/last.ckpt")
    artifact = export_inference_artifact(run, tmp_path / "artifact")
    for loader in (
        {"_target_": "quick_convert.inference.load_inference_system", "source": str(run)},
        {"_target_": "quick_convert.inference.load_system_from_run", "run_dir": str(run)},
        {"_target_": "quick_convert.inference.load_inference_system", "source": str(run / "checkpoints/last.ckpt")},
        {"_target_": "quick_convert.inference.load_inference_artifact", "artifact_dir": str(artifact)},
    ):
        config = OmegaConf.create({"system": loader, "dataset": "???"})
        plan = dependency_plan(config)
        assert plan.extras == ("transformers",)
        assert "does.not.exist.Trainer" not in plan.targets
        override_devices(config.system, "cpu")
        assert config.system.map_location == "cpu"


def test_existing_system_bases_inherit_the_shared_loader():
    from quick_convert.systems.anonymization import BaseAnonymizer
    from quick_convert.systems.asr import VQASRSystem
    from quick_convert.systems.asr.base import ASRSystem
    from quick_convert.systems.reconstruction import SSLReconstructionSystem
    from quick_convert.systems.ser.base import SERSystem

    for cls in (BaseAnonymizer, ASRSystem, VQASRSystem, SSLReconstructionSystem, SERSystem):
        assert cls.from_run.__func__ is InferenceSystem.from_run.__func__


def test_actual_training_checkpoint_is_ready_for_direct_inference(tmp_path):
    pytest.importorskip("lightning")
    from quick_convert.pipelines.training.pipeline import TrainingPipeline
    from quick_convert.training.lightning.modules.base import BaseTrainingModule
    from quick_convert.training.lightning.optim import Optimization
    from quick_convert.training.lightning.trainer import LightningTrainer

    class TrainingWrapper(BaseTrainingModule):
        def __init__(self):
            super().__init__(
                Optimization(optimizer_kwargs={"lr": 0.01}, lr_scheduler=None), enable_grad_norm_logging=False
            )
            self.system = NewTask()
            self.training_only = nn.Parameter(torch.ones(1))

        def _shared_step(self, batch, stage):
            return SimpleNamespace(loss=self.system(batch).square().mean())

        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            return batch.to(device)

    class Dataset:
        def make_dataloader(self, **kwargs):
            return torch.utils.data.DataLoader(torch.ones(4, 2), batch_size=2)

    module = TrainingWrapper()
    trainer = LightningTrainer(
        module,
        trainer_kwargs=dict(
            max_steps=1,
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        ),
    )
    pipeline = TrainingPipeline(trainer, Dataset(), out_dir=tmp_path / "run")
    pipeline.prepare()
    pipeline.write_config(OmegaConf.to_yaml({"system": task_config(), "training": {"_target_": "missing.Trainer"}}))
    pipeline.run()
    path = tmp_path / "standalone.ckpt"
    trainer.pl_trainer.save_checkpoint(path)
    saved = torch.load(path, weights_only=True)
    assert saved[CHECKPOINT_METADATA_KEY]["system"] == task_config()
    assert "training" not in saved[CHECKPOINT_METADATA_KEY]
    assert saved["global_step"] == 1
    loaded = load_inference_system(path)
    torch.testing.assert_close(loaded(torch.ones(1, 2)), module.system(torch.ones(1, 2)))
    assert not hasattr(loaded, "training_only")


@pytest.mark.parametrize("prefix", ["system.online_encoders", "system.online_encoders.content.weight"])
def test_checkpoint_metadata_is_independent_and_refuses_omitted_trainable_encoders(prefix):
    pytest.importorskip("lightning")
    from quick_convert.training.lightning.modules.base import BaseTrainingModule

    class Wrapper(BaseTrainingModule):
        def _shared_step(self, batch, stage):
            pass

    module = Wrapper(optimization=object(), checkpoint_exclude_prefixes=(prefix,))
    module.system = NewTask()
    module.system.online_encoders = nn.ModuleDict({"content": nn.Linear(2, 2)})
    module.set_inference_config({"system": task_config()})
    checkpoint = {"state_dict": module.state_dict()}
    with pytest.raises(ValueError, match="Cannot omit trainable"):
        module.on_save_checkpoint(checkpoint)
    module.system.online_encoders.requires_grad_(False)
    module.on_save_checkpoint(checkpoint)
    assert "system.online_encoders.content.weight" not in checkpoint["state_dict"]
    checkpoint[CHECKPOINT_METADATA_KEY]["system"]["dimension"] = 9
    assert module._inference_description["system"]["dimension"] == 2
