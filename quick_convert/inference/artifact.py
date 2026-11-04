from __future__ import annotations

from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from quick_convert.utils.device import DeviceLike


ARTIFACT_FORMAT = "quick-convert-inference"
ARTIFACT_VERSION = 1
MANIFEST_NAME = "artifact.yaml"
WEIGHTS_NAME = "weights.pt"


def _normalize_prefix(prefix: str) -> str:
    return prefix.rstrip(".") + "."


def _matches_prefix(key: str, prefixes: tuple[str, ...]) -> bool:
    return any(key == prefix.rstrip(".") or key.startswith(prefix) for prefix in prefixes)


def _prepare_destination(destination: PathLike, *, overwrite: bool) -> Path:
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"Inference artifact directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _resolved_system_config(system_config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    config = system_config if isinstance(system_config, DictConfig) else OmegaConf.create(system_config)
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict) or "_target_" not in resolved:
        raise ValueError("An inference artifact requires a Hydra system config with `_target_`.")
    return resolved


def _write_artifact(
    destination: PathLike,
    *,
    system_config: Mapping[str, Any] | DictConfig,
    state_dict: Mapping[str, torch.Tensor],
    excluded_state_prefixes: Iterable[str] = (),
    overwrite: bool = False,
) -> Path:
    destination = _prepare_destination(destination, overwrite=overwrite)
    prefixes = sorted({_normalize_prefix(prefix) for prefix in excluded_state_prefixes})
    manifest = {
        "format": ARTIFACT_FORMAT,
        "version": ARTIFACT_VERSION,
        "weights": WEIGHTS_NAME,
        "excluded_state_prefixes": prefixes,
        "system": _resolved_system_config(system_config),
    }

    torch.save(dict(state_dict), destination / WEIGHTS_NAME)
    OmegaConf.save(OmegaConf.create(manifest), destination / MANIFEST_NAME)
    return destination


def save_inference_artifact(
    system: nn.Module,
    system_config: Mapping[str, Any] | DictConfig,
    destination: PathLike,
    *,
    excluded_state_prefixes: Iterable[str] = (),
    overwrite: bool = False,
) -> Path:
    """Save an instantiated system as a portable inference artifact.

    The artifact contains a resolved Hydra system configuration and CPU model
    weights. It intentionally contains no training-module, optimizer, logger,
    scheduler, or callback state.

    Args:
        system: Instantiated inference system whose state should be saved.
        system_config: Resolvable Hydra configuration for ``system``. It must
            contain a ``_target_`` key.
        destination: Directory in which to write ``artifact.yaml`` and
            ``weights.pt``.
        excluded_state_prefixes: State-dict prefixes to omit. Use this only for
            reproducible external state, such as a pretrained online encoder.
        overwrite: Allow writing into a non-empty destination directory.

    Returns:
        The artifact directory.

    Raises:
        FileExistsError: If the destination is non-empty and overwrite is false.
        ValueError: If the system configuration is not a resolved Hydra target.
    """
    excluded = tuple(_normalize_prefix(prefix) for prefix in excluded_state_prefixes)
    state_dict = {
        key: value.detach().cpu() for key, value in system.state_dict().items() if not _matches_prefix(key, excluded)
    }
    return _write_artifact(
        destination,
        system_config=system_config,
        state_dict=state_dict,
        excluded_state_prefixes=excluded,
        overwrite=overwrite,
    )


def _system_checkpoint_state(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    raw_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state, Mapping):
        raise TypeError("Checkpoint `state_dict` must be a mapping.")

    cleaned = {key.replace("._orig_mod.", "."): value for key, value in raw_state.items()}
    system_state = {
        key.removeprefix("system."): value.detach().cpu()
        for key, value in cleaned.items()
        if key.startswith("system.") and isinstance(value, torch.Tensor)
    }
    if not system_state:
        if not all(isinstance(value, torch.Tensor) for value in cleaned.values()):
            raise ValueError("Checkpoint does not contain `system.*` weights or a plain system state dict.")
        system_state = {key: value.detach().cpu() for key, value in cleaned.items()}

    checkpoint_prefixes = checkpoint.get("checkpoint_exclude_prefixes", ())
    excluded = tuple(
        _normalize_prefix(prefix.removeprefix("system."))
        for prefix in checkpoint_prefixes
        if prefix.startswith("system.")
    )
    return system_state, excluded


def export_inference_artifact(
    run_dir: PathLike,
    destination: PathLike,
    *,
    checkpoint: PathLike = "checkpoints/last.ckpt",
    config: PathLike | None = None,
    map_location: DeviceLike = "cpu",
    overwrite: bool = False,
) -> Path:
    """Export the inference system from a Quick Convert training run.

    Embedded checkpoint metadata supplies the system definition when available;
    otherwise the run config supplies top-level ``system``. For compatibility,
    older runs may use ``architecture.system``. Lightning
    checkpoints are reduced to ``system.*`` parameters; plain system state
    dictionaries are also accepted.

    Args:
        run_dir: Directory containing the resolved run config and checkpoint.
        destination: Output artifact directory.
        checkpoint: Absolute checkpoint path or path relative to ``run_dir``.
        config: Explicit YAML config override, relative to ``run_dir`` by default.
            If omitted, use embedded metadata or fall back to config.yaml.
        map_location: Device used while reading the checkpoint.
        overwrite: Allow writing into a non-empty destination directory.

    Returns:
        The artifact directory.

    Raises:
        ValueError: If the run has no system config or checkpoint weights.
        TypeError: If the checkpoint is not a supported state mapping.
    """
    from .source import read_run_state

    state = read_run_state(run_dir, checkpoint=checkpoint, config=config, map_location=map_location)
    return _write_artifact(
        destination,
        system_config=state.system_config,
        state_dict=state.state_dict,
        excluded_state_prefixes=state.excluded_state_prefixes,
        overwrite=overwrite,
    )


def load_inference_artifact(
    artifact_dir: PathLike,
    *,
    map_location: DeviceLike = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Instantiate a system and load a versioned inference artifact.

    Device fields in the saved system configuration are overridden with
    ``map_location`` before Hydra instantiation. The returned module is moved
    to that device, frozen, and placed in evaluation mode.

    Args:
        artifact_dir: Directory containing ``artifact.yaml`` and ``weights.pt``.
        map_location: Device on which to instantiate and load the system.
        strict: Forwarded to :meth:`torch.nn.Module.load_state_dict`. Deliberately
            excluded prefixes are restored from the fresh system first.

    Returns:
        The configured inference system in evaluation mode.

    Raises:
        ValueError: If the artifact format or version is unsupported.
    """
    artifact_dir = Path(artifact_dir).expanduser()
    manifest = OmegaConf.load(artifact_dir / MANIFEST_NAME)
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"Unsupported inference artifact format: {manifest.get('format')!r}")
    if manifest.get("version") != ARTIFACT_VERSION:
        raise ValueError(f"Unsupported inference artifact version: {manifest.get('version')!r}")

    from .source import InferenceState, restore_system

    state = InferenceState(
        system_config=OmegaConf.to_container(manifest.system, resolve=True),
        state_dict=torch.load(artifact_dir / manifest.weights, map_location="cpu", weights_only=True),
        excluded_state_prefixes=tuple(manifest.get("excluded_state_prefixes", ())),
    )
    return restore_system(state, map_location=map_location, strict=strict)
