"""Plan optional dependencies from a composed Quick Convert object graph."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, OmegaConf

from quick_convert.utils.resolvers import register_config_resolvers


_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
_REGISTRY_PATH = _CONFIG_DIR / "dependencies.yaml"


@dataclass(frozen=True)
class DependencyPlan:
    """Optional extras required by the targets in one composed run."""

    extras: tuple[str, ...]
    targets: tuple[str, ...]
    probes: dict[str, tuple[str, ...]]

    @property
    def uv_command(self) -> str:
        """Return the corresponding uv installation command."""
        suffix = " ".join(f"--extra {extra}" for extra in self.extras)
        return f"uv sync {suffix}".rstrip()

    @property
    def pip_specifier(self) -> str:
        """Return the corresponding published-package requirement."""
        return f"quick-convert[{','.join(self.extras)}]" if self.extras else "quick-convert"

    def missing_extras(self) -> tuple[str, ...]:
        """Return extras with at least one unavailable import probe."""
        return tuple(
            extra
            for extra in self.extras
            if any(not _module_available(module) for module in self.probes.get(extra, ()))
        )


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError):
        return False


def _targets(value: Any) -> set[str]:
    targets: set[str] = set()
    if isinstance(value, dict):
        target = value.get("_target_")
        if isinstance(target, str):
            targets.add(target)
        for child in value.values():
            targets.update(_targets(child))
    elif isinstance(value, list):
        for child in value:
            targets.update(_targets(child))
    return targets


def _inference_targets(config: DictConfig | ListConfig) -> set[str]:
    """Inspect saved compositions without constructing their optional backends."""
    targets = set()
    if isinstance(config, DictConfig):
        loader = config.get("_target_")
        if loader in {"quick_convert.inference.load_inference_system", "quick_convert.inference.load_system_from_run"}:
            from .inference.source import inference_system_config

            field = "source" if loader.endswith("load_inference_system") else "run_dir"
            return _targets(
                inference_system_config(config[field], checkpoint=config.get("checkpoint"), config=config.get("config"))
            )
        if loader == "quick_convert.inference.load_inference_artifact":
            from .inference.source import inference_system_config

            return _targets(inference_system_config(config.artifact_dir))
    # Follow concrete nested mappings, leaving unrelated aliases and missing
    # workflow fields unresolved, as in the ordinary dependency graph walk.
    raw = OmegaConf.to_container(config, resolve=False)
    values = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for key, value in values:
        if isinstance(value, (dict, list)):
            targets.update(_inference_targets(config[key]))
    return targets


def dependency_plan(config: DictConfig) -> DependencyPlan:
    """Build a dependency plan from an already composed configuration."""
    registry = OmegaConf.to_container(OmegaConf.load(_REGISTRY_PATH), resolve=True)
    graph = OmegaConf.to_container(config, resolve=False)
    if not isinstance(registry, dict) or not isinstance(graph, dict):
        raise TypeError("Dependency registry and composed config must be mappings.")

    targets = sorted(_targets(graph) | _inference_targets(config))
    target_extras = registry.get("targets", {})
    extra_metadata = registry.get("extras", {})
    extras = sorted({extra for target in targets for extra in target_extras.get(target, [])})
    probes = {extra: tuple(extra_metadata.get(extra, {}).get("probes", ())) for extra in extras}
    return DependencyPlan(tuple(extras), tuple(targets), probes)


def compose_dependency_plan(run_config: str, overrides: list[str] | None = None) -> DependencyPlan:
    """Compose a packaged run and return its optional dependency plan."""
    register_config_resolvers()
    config_name = run_config.removesuffix(".yaml")
    if not config_name.startswith("run/"):
        config_name = f"run/{config_name}"
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        config = compose(
            config_name=config_name,
            overrides=list(overrides or ()),
        )
    return dependency_plan(config)
