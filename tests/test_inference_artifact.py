from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from quick_convert.inference import (
    ARTIFACT_FORMAT,
    ARTIFACT_VERSION,
    export_inference_artifact,
    load_inference_artifact,
    save_inference_artifact,
)


LINEAR_CONFIG = {
    "_target_": "torch.nn.Linear",
    "in_features": 3,
    "out_features": 2,
}
CONFIG_DIR = Path(__file__).parents[1] / "configs"


def register_config_resolvers():
    resolvers = {
        "add": lambda x, y: int(x) + int(y),
        "mul": lambda x, y: int(x) * int(y),
        "floor": lambda x, y: int(int(x) / int(y)),
        "bool": bool,
        "len": len,
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)


def test_plain_system_artifact_round_trips(tmp_path):
    system = nn.Linear(3, 2)
    with torch.no_grad():
        system.weight.fill_(2.0)
        system.bias.fill_(3.0)

    artifact_dir = save_inference_artifact(system, LINEAR_CONFIG, tmp_path / "artifact")
    loaded = load_inference_artifact(artifact_dir)
    manifest = OmegaConf.load(artifact_dir / "artifact.yaml")

    assert manifest.format == ARTIFACT_FORMAT
    assert manifest.version == ARTIFACT_VERSION
    assert manifest.system._target_ == "torch.nn.Linear"
    assert not loaded.training
    torch.testing.assert_close(loaded.weight, system.weight)
    torch.testing.assert_close(loaded.bias, system.bias)


def test_training_run_export_keeps_only_system_state(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    OmegaConf.save(
        OmegaConf.create({"system": LINEAR_CONFIG}),
        run_dir / "config.yaml",
    )
    checkpoint = {
        "state_dict": {
            "system.weight": torch.full((2, 3), 4.0),
            "system.bias": torch.full((2,), 5.0),
            "training_only": torch.tensor(6.0),
        }
    }
    torch.save(checkpoint, checkpoint_dir / "last.ckpt")

    artifact_dir = export_inference_artifact(run_dir, tmp_path / "artifact")
    saved_state = torch.load(artifact_dir / "weights.pt", weights_only=True)
    loaded = load_inference_artifact(artifact_dir)

    assert saved_state.keys() == {"weight", "bias"}
    torch.testing.assert_close(loaded.weight, checkpoint["state_dict"]["system.weight"])
    torch.testing.assert_close(loaded.bias, checkpoint["state_dict"]["system.bias"])


def test_export_accepts_legacy_architecture_system_config(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    OmegaConf.save(
        OmegaConf.create({"architecture": {"system": LINEAR_CONFIG}}),
        run_dir / "config.yaml",
    )
    torch.save(
        {"state_dict": {"system.weight": torch.ones(2, 3), "system.bias": torch.ones(2)}},
        checkpoint_dir / "last.ckpt",
    )

    artifact_dir = export_inference_artifact(run_dir, tmp_path / "artifact")

    assert (artifact_dir / "artifact.yaml").is_file()


def test_run_export_resolves_system_config_in_its_full_hydra_context(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    register_config_resolvers()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        config = compose(config_name="run/train_sslr_w2vbert_cmdiff")
    resolved_system = OmegaConf.to_container(config.system, resolve=True)
    OmegaConf.save(
        OmegaConf.create({"system": resolved_system}),
        run_dir / "config.yaml",
    )
    torch.save(
        {"state_dict": {"system.placeholder": torch.tensor(1.0)}},
        checkpoint_dir / "last.ckpt",
    )

    artifact_dir = export_inference_artifact(run_dir, tmp_path / "artifact")
    manifest = OmegaConf.load(artifact_dir / "artifact.yaml")

    assert manifest.system.online_encoders.content.device == config.device
    assert manifest.system.decoder.feature_dim == 1024
    assert manifest.system.decoder.flow.spk_embed_dim == 192


def test_run_export_accepts_a_plain_system_state_dict(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    OmegaConf.save(
        OmegaConf.create({"system": LINEAR_CONFIG}),
        run_dir / "config.yaml",
    )
    torch.save(
        {
            "weight": torch.full((2, 3), 8.0),
            "bias": torch.full((2,), 9.0),
        },
        checkpoint_dir / "last.ckpt",
    )

    artifact_dir = export_inference_artifact(run_dir, tmp_path / "artifact")
    loaded = load_inference_artifact(artifact_dir)

    torch.testing.assert_close(loaded.weight, torch.full((2, 3), 8.0))
    torch.testing.assert_close(loaded.bias, torch.full((2,), 9.0))


def test_declared_external_weights_are_reconstructed_before_strict_loading(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    OmegaConf.save(
        OmegaConf.create({"system": LINEAR_CONFIG}),
        run_dir / "config.yaml",
    )
    torch.save(
        {
            "state_dict": {"system.weight": torch.full((2, 3), 7.0)},
            "checkpoint_exclude_prefixes": ["system.bias."],
        },
        checkpoint_dir / "last.ckpt",
    )

    artifact_dir = export_inference_artifact(run_dir, tmp_path / "artifact")
    loaded = load_inference_artifact(artifact_dir, strict=True)
    manifest = OmegaConf.load(artifact_dir / "artifact.yaml")

    assert list(manifest.excluded_state_prefixes) == ["bias."]
    torch.testing.assert_close(loaded.weight, torch.full((2, 3), 7.0))


def test_artifact_refuses_to_overwrite_a_nonempty_directory(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "keep.txt").write_text("keep")

    with pytest.raises(FileExistsError, match="not empty"):
        save_inference_artifact(nn.Linear(3, 2), LINEAR_CONFIG, artifact_dir)


def test_loader_rejects_unknown_versions(tmp_path):
    artifact_dir = save_inference_artifact(nn.Linear(3, 2), LINEAR_CONFIG, tmp_path / "artifact")
    manifest_path = artifact_dir / "artifact.yaml"
    manifest = OmegaConf.load(manifest_path)
    manifest.version = ARTIFACT_VERSION + 1
    OmegaConf.save(manifest, manifest_path)

    with pytest.raises(ValueError, match="version"):
        load_inference_artifact(artifact_dir)
