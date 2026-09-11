from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def _register_resolvers() -> None:
    resolvers = {
        "add": lambda x, y: int(x) + int(y),
        "mul": lambda x, y: int(x) * int(y),
        "floor": lambda x, y: int(int(x) / int(y)),
        "bool": lambda value: bool(value),
        "len": len,
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)


_register_resolvers()


def execute_pipeline(cfg: DictConfig) -> Any:
    """Instantiate and execute the pipeline described by a composed config."""
    rendered_config = OmegaConf.to_yaml(cfg, resolve=True)
    print(rendered_config)

    pipeline = instantiate(cfg.pipeline)
    write_config = getattr(pipeline, "write_config", None)
    if callable(write_config):
        write_config(rendered_config)

    run_kwargs = cfg.get("run", {})
    if not isinstance(run_kwargs, Mapping):
        raise TypeError("The optional top-level `run` config must be a mapping.")
    return pipeline.run(**run_kwargs)


@hydra.main(version_base=None, config_path="../../configs", config_name=None)
def main(cfg: DictConfig) -> Any:
    return execute_pipeline(cfg)


if __name__ == "__main__":
    main()
