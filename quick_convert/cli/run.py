from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from quick_convert.utils.resolvers import register_config_resolvers


def _register_resolvers() -> None:
    register_config_resolvers()


_register_resolvers()


def execute_pipeline(cfg: DictConfig) -> Any:
    """Instantiate and execute the pipeline described by a composed config."""
    rendered_config = OmegaConf.to_yaml(cfg, resolve=True)
    print(rendered_config)

    pipeline = instantiate(cfg.pipeline)
    prepare = getattr(pipeline, "prepare", None)
    if callable(prepare):
        prepare()

    write_config = getattr(pipeline, "write_config", None)
    if callable(write_config):
        write_config(rendered_config)

    run_kwargs = cfg.get("run", {})
    if not isinstance(run_kwargs, Mapping):
        raise TypeError("The optional top-level `run` config must be a mapping.")
    return pipeline.run(**run_kwargs)


@hydra.main(version_base=None, config_path="../configs", config_name=None)
def main(cfg: DictConfig) -> Any:
    return execute_pipeline(cfg)


if __name__ == "__main__":
    main()
