import importlib
from copy import deepcopy

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from quick_convert.inference import load_inference_system, save_inference_artifact
from quick_convert.inference.migration import migrate_system_config
from quick_convert.inference.source import CHECKPOINT_METADATA_KEY, SOURCE_FORMAT, SOURCE_VERSION
from quick_convert.systems.conversion import SSLReconstructionConversionSystem


def composition():
    return {
        "_target_": "quick_convert.systems.reconstruction.ssl.SSLReconstructionSystem",
        "decoder": {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 2},
    }


@pytest.mark.parametrize("source_kind", ["run", "checkpoint", "artifact"])
def test_old_ssl_compositions_restore_identical_weights(tmp_path, source_kind):
    original = SSLReconstructionConversionSystem(nn.Linear(2, 2))
    config = composition()
    if source_kind == "artifact":
        source = save_inference_artifact(original, config, tmp_path / "artifact")
        manifest = OmegaConf.load(source / "artifact.yaml")
        manifest.system._target_ = config["_target_"]
        OmegaConf.save(manifest, source / "artifact.yaml")
    else:
        (tmp_path / "checkpoints").mkdir()
        saved = {"state_dict": {"system." + key: value for key, value in original.state_dict().items()}}
        OmegaConf.save(OmegaConf.create({"system": config}), tmp_path / "config.yaml")
        if source_kind == "checkpoint":
            saved[CHECKPOINT_METADATA_KEY] = {"format": SOURCE_FORMAT, "version": SOURCE_VERSION, "system": config}
        path = tmp_path / "checkpoints/last.ckpt"
        torch.save(saved, path)
        source = path if source_kind == "checkpoint" else tmp_path
    loaded = load_inference_system(source)
    assert isinstance(loaded, SSLReconstructionConversionSystem)
    assert loaded.state_dict().keys() == original.state_dict().keys()
    torch.testing.assert_close(loaded.decoder(torch.ones(1, 2)), original.decoder(torch.ones(1, 2)))
    assert not loaded.training


@pytest.mark.parametrize(
    "old_target",
    [
        "quick_convert.systems.reconstruction.ssl.SSLReconstructionSystem",
        "quick_convert.systems.conversion.ssl.SSLReconstructionConversionSystem",
    ],
)
def test_migration_copies_and_changes_only_known_target_fields(old_target):
    config = {"nested": [{"_target_": old_target}], "note": old_target}
    before = deepcopy(config)
    migrated = migrate_system_config(config)
    assert config == before
    assert migrated["note"] == config["note"]
    assert migrated["nested"][0]["_target_"] == (
        "quick_convert.systems.conversion.ssl_reconstruction.SSLReconstructionConversionSystem"
    )


@pytest.mark.parametrize(
    "module",
    [
        "quick_convert.systems.anonymization",
        "quick_convert.systems.reconstruction",
        "quick_convert.pipelines.anonymization",
    ],
)
def test_retired_hierarchies_are_absent(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.mark.parametrize(
    "old_class,new_class", [("KNNVCAnonymizer", "KNNVCConversionSystem"), ("ASRBNAnonymizer", "ASRBNConversionSystem")]
)
def test_old_hub_backend_targets_restore_weights_and_override_device(monkeypatch, old_class, new_class):
    from quick_convert.inference.source import InferenceState, restore_system
    from quick_convert.systems import conversion

    model = nn.Linear(2, 2)
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return nn.Linear(2, 2)

    monkeypatch.setattr(torch.hub, "load", load)
    state = InferenceState(
        {"_target_": f"quick_convert.systems.anonymization.{old_class}"},
        {"model." + k: v for k, v in model.state_dict().items()},
    )
    restored = restore_system(state, map_location="cpu")
    assert isinstance(restored, getattr(conversion, new_class))
    torch.testing.assert_close(restored.model.weight, model.weight)
    if old_class == "KNNVCAnonymizer":
        assert calls[0]["device"] == "cpu"


def test_historical_training_yaml_migrates_nested_targets():
    saved = OmegaConf.create("""
    trainer:
      _target_: quick_convert.pipelines.training.lightning_trainer.LightningTrainer
      module:
        _target_: quick_convert.pipelines.training.modules.vq_asr.VQASRTrainingModule
        optimization:
          _target_: quick_convert.pipelines.training.optim.base.Optimization
          warmup:
            _target_: quick_convert.pipelines.training.optim.base.LinearWarmup
    reconstruction:
      _target_: quick_convert.pipelines.training.modules.ssl_reconstruction.SSLReconstructionTrainingModule
    tokenizer_trainer:
      _target_: quick_convert.pipelines.training.sentencepiece_trainer.TokenizerTrainer
      tokenizer:
        _target_: quick_convert.pipelines.training.modules.tokenizer.bpe.SentencePieceBPETrainer
    """)
    migrated = OmegaConf.create(migrate_system_config(OmegaConf.to_container(saved)))
    assert migrated.trainer._target_ == "quick_convert.training.lightning.trainer.LightningTrainer"
    assert migrated.trainer.module._target_ == "quick_convert.training.lightning.modules.vq_asr.VQASRTrainingModule"
    assert migrated.trainer.module.optimization._target_ == "quick_convert.training.lightning.optim.Optimization"
    assert migrated.trainer.module.optimization.warmup._target_ == "quick_convert.training.lightning.optim.LinearWarmup"
    assert migrated.reconstruction._target_ == (
        "quick_convert.training.lightning.modules.ssl_reconstruction.SSLReconstructionTrainingModule"
    )
    assert migrated.tokenizer_trainer._target_ == "quick_convert.training.tokenizer.TokenizerTrainer"
    assert migrated.tokenizer_trainer.tokenizer._target_ == (
        "quick_convert.training.tokenizers.sentencepiece.SentencePieceBPETrainer"
    )
