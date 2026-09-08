from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict


_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def compose_component(
    group: str,
    name: str,
    overrides: Mapping[str, Any] | None = None,
) -> DictConfig:
    with initialize_config_dir(
        config_dir=str(_CONFIG_DIR),
        version_base=None,
    ):
        cfg = compose(
            config_name="library",
            overrides=[f"+{group}@component={name}"],
        )

    component = cfg.component

    if overrides:
        with open_dict(component):
            component.merge_with(OmegaConf.create(dict(overrides)))

    return component
