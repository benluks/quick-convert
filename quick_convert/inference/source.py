"""Restore inference systems without constructing their training wrappers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from quick_convert.utils.device import DeviceLike, configure_device, override_devices
from quick_convert.utils.paths import checkpoint_run_config
from quick_convert.utils.resolvers import register_config_resolvers

from .migration import migrate_system_config


SOURCE_FORMAT = "quick-convert-inference-source"
SOURCE_VERSION = 1
CHECKPOINT_METADATA_KEY = "quick_convert_inference"


@dataclass
class InferenceState:
    """Resolved system composition, tensor state, and declared omissions."""

    system_config: dict[str, Any]
    state_dict: dict[str, torch.Tensor]
    excluded_state_prefixes: tuple[str, ...] = ()


def inference_description(config: Mapping) -> dict | None:
    """Resolve just the inference system in the full run's interpolation context.

    Training-only compositions return None. The older architecture.system
    location remains readable for existing runs.
    """
    register_config_resolvers()
    cfg = OmegaConf.create(config)
    system = OmegaConf.select(cfg, "system")
    if system is None:
        system = OmegaConf.select(cfg, "architecture.system")
    if system is None:
        return None
    resolved = OmegaConf.to_container(system, resolve=True)
    if not isinstance(resolved, dict) or "_target_" not in resolved:
        raise ValueError("Inference requires a Hydra system config with `_target_`.")
    return {"format": SOURCE_FORMAT, "version": SOURCE_VERSION, "system": migrate_system_config(resolved)}


def _state_from_description(description: Mapping, checkpoint: Mapping) -> InferenceState:
    if not isinstance(description, Mapping):
        raise ValueError("Checkpoint inference metadata must be a mapping.")
    if description.get("format") != SOURCE_FORMAT or description.get("version") != SOURCE_VERSION:
        raise ValueError("Unsupported checkpoint inference metadata format or version.")
    config = description.get("system")
    if not isinstance(config, Mapping) or "_target_" not in config:
        raise ValueError("Checkpoint inference metadata is missing its system composition.")
    if description.get("adapter") is not None:
        raise ValueError("Checkpoint requires an unsupported inference adapter.")
    from .artifact import _system_checkpoint_state

    weights, exclusions = _system_checkpoint_state(checkpoint)
    return InferenceState(migrate_system_config(config), weights, exclusions)


def _run_description(path: Path) -> dict:
    description = inference_description(OmegaConf.load(path))
    if description is None:
        raise ValueError("Run config does not define `system`.")
    return description


def read_run_state(
    run_dir: PathLike,
    *,
    checkpoint: PathLike = "checkpoints/last.ckpt",
    config: PathLike | None = None,
    map_location: DeviceLike = "cpu",
) -> InferenceState:
    """Read a run on CPU by default, without instantiation or filesystem writes.

    Embedded inference metadata takes precedence over config.yaml. An explicit
    config selects that composition instead, including for newer checkpoints.
    """
    run_dir = Path(run_dir).expanduser()
    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
        path = run_dir / path
    saved = torch.load(path, map_location=configure_device(map_location), weights_only=False)
    if not isinstance(saved, Mapping):
        raise TypeError("Training checkpoint must be a mapping.")
    description = saved.get(CHECKPOINT_METADATA_KEY) if config is None else None
    if description is None:
        config_path = Path(config).expanduser() if config is not None else Path("config.yaml")
        description = _run_description(run_dir / config_path)
    return _state_from_description(description, saved)


def restore_system(state: InferenceState, *, map_location: DeviceLike = "cpu", strict: bool = True) -> nn.Module:
    """Instantiate, restore, and freeze the same system for all source formats."""
    device = configure_device(map_location)
    config = OmegaConf.create(migrate_system_config(state.system_config))
    override_devices(config, str(device))
    system = instantiate(config)
    if not isinstance(system, nn.Module):
        raise TypeError("Inference system composition must instantiate a torch.nn.Module.")
    system = system.to(device)
    weights = dict(state.state_dict)
    from .artifact import _matches_prefix, _normalize_prefix

    prefixes = tuple(_normalize_prefix(prefix) for prefix in state.excluded_state_prefixes)
    if prefixes:
        for key, value in system.state_dict().items():
            if _matches_prefix(key, prefixes):
                weights.setdefault(key, value)
    system.load_state_dict(weights, strict=strict)
    return system.eval().requires_grad_(False)


def load_system_from_run(
    run_dir: PathLike,
    *,
    checkpoint: PathLike = "checkpoints/last.ckpt",
    config: PathLike | None = None,
    map_location: DeviceLike = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Load only the inference system saved in a run directory.

    Checkpoint and config paths are relative to run_dir unless absolute.
    No training module, optimizer, logger, or intermediate artifact is created.
    """
    return _record_source(
        restore_system(
            read_run_state(run_dir, checkpoint=checkpoint, config=config), map_location=map_location, strict=strict
        ),
        run_dir,
        checkpoint,
        config,
    )


def _record_source(system, source, checkpoint=None, config=None):
    """Retain loader provenance for workflows without attaching training state."""
    system._inference_source = {"source": str(Path(source).expanduser().resolve())}
    if checkpoint is not None:
        system._inference_source["checkpoint"] = str(checkpoint)
    if config is not None:
        system._inference_source["config"] = str(config)
    return system


def _artifact_config(source: Path, checkpoint: PathLike | None, config: PathLike | None) -> dict | None:
    from .artifact import ARTIFACT_FORMAT, ARTIFACT_VERSION, MANIFEST_NAME

    path = source / MANIFEST_NAME
    if not path.is_file():
        return None
    if checkpoint is not None or config is not None:
        raise ValueError("Exported artifacts already select their weights and system config; omit checkpoint/config.")
    manifest = OmegaConf.load(path)
    if manifest.get("format") != ARTIFACT_FORMAT or manifest.get("version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported inference artifact format or version.")
    return migrate_system_config(OmegaConf.to_container(manifest.system, resolve=True))


def _read_training_source(source: Path, checkpoint: PathLike | None, config: PathLike | None) -> InferenceState:
    if source.is_file():
        if checkpoint is not None:
            raise ValueError("checkpoint selects a file inside a run; source already names a checkpoint file.")
        if config is not None:
            return read_run_state(source.parent, checkpoint=source.resolve(), config=config)
        saved = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(saved, Mapping):
            raise TypeError("Training checkpoint must be a mapping.")
        if CHECKPOINT_METADATA_KEY in saved:
            return _state_from_description(saved[CHECKPOINT_METADATA_KEY], saved)
        discovered = checkpoint_run_config(source)
        if discovered is None:
            raise ValueError(
                "Checkpoint has no inference composition or nearby run config.yaml; pass config explicitly."
            )
        return _state_from_description(_run_description(discovered), saved)
    return read_run_state(
        source, checkpoint=checkpoint if checkpoint is not None else "checkpoints/last.ckpt", config=config
    )


def inference_system_config(
    source: PathLike, *, checkpoint: PathLike | None = None, config: PathLike | None = None
) -> dict:
    """Inspect the saved system composition without importing its backends.

    Run/checkpoint inspection reads checkpoint metadata and tensors on CPU;
    artifact inspection reads only its manifest.
    """
    source = Path(source).expanduser()
    artifact = _artifact_config(source, checkpoint, config)
    if artifact is not None:
        return artifact
    return _read_training_source(source, checkpoint, config).system_config


def load_inference_system(
    source: PathLike,
    *,
    checkpoint: PathLike | None = None,
    config: PathLike | None = None,
    map_location: DeviceLike = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Load a run directory, exported artifact, or standalone checkpoint.

    Older checkpoint files need a nearby config.yaml or an explicit config.
    New checkpoints embed inference composition through the training pipeline.
    Loading returns a frozen module in evaluation mode on map_location.
    """
    source = Path(source).expanduser()
    if _artifact_config(source, checkpoint, config) is not None:
        from .artifact import load_inference_artifact

        return load_inference_artifact(source, map_location=map_location, strict=strict)
    return _record_source(
        restore_system(_read_training_source(source, checkpoint, config), map_location=map_location, strict=strict),
        source,
        checkpoint,
        config,
    )
