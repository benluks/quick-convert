from __future__ import annotations

from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn

from quick_convert.utils.device import DeviceLike, configure_device, override_devices


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
    """Save an instantiated system as a portable inference artifact."""
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
        if not cleaned or not all(isinstance(value, torch.Tensor) for value in cleaned.values()):
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
    config: PathLike = "config.yaml",
    map_location: DeviceLike = "cpu",
    overwrite: bool = False,
) -> Path:
    """Export ``architecture.system`` from a Lightning training run."""
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / config)
    system_config = OmegaConf.select(cfg, "architecture.system")
    if system_config is None:
        raise ValueError("Run config does not define `architecture.system`.")

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path
    checkpoint_state = torch.load(
        checkpoint_path,
        map_location=configure_device(map_location),
        weights_only=False,
    )
    if not isinstance(checkpoint_state, Mapping):
        raise TypeError("Training checkpoint must be a mapping.")
    state_dict, excluded = _system_checkpoint_state(checkpoint_state)

    return _write_artifact(
        destination,
        system_config=system_config,
        state_dict=state_dict,
        excluded_state_prefixes=excluded,
        overwrite=overwrite,
    )


def load_inference_artifact(
    artifact_dir: PathLike,
    *,
    map_location: DeviceLike = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Instantiate and load a versioned inference artifact."""
    artifact_dir = Path(artifact_dir)
    manifest = OmegaConf.load(artifact_dir / MANIFEST_NAME)
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"Unsupported inference artifact format: {manifest.get('format')!r}")
    if manifest.get("version") != ARTIFACT_VERSION:
        raise ValueError(f"Unsupported inference artifact version: {manifest.get('version')!r}")

    device = configure_device(map_location)
    system_config = manifest.system
    override_devices(system_config, str(device))
    system = instantiate(system_config).to(device)

    weights_path = artifact_dir / manifest.weights
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    excluded = tuple(_normalize_prefix(prefix) for prefix in manifest.get("excluded_state_prefixes", ()))
    if excluded:
        for key, value in system.state_dict().items():
            if _matches_prefix(key, excluded):
                state_dict.setdefault(key, value)

    system.load_state_dict(state_dict, strict=strict)
    system.eval()
    return system
