from __future__ import annotations

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))


@hydra.main(version_base=None, config_path="../../configs", config_name="run/test")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    for key, value in cfg.items():
        if isinstance(value, DictConfig) and "_target_" in value:
            print(f"\nInstantiating: {key}")
            obj = instantiate(value)
            globals()[key] = obj
            print(obj)


if __name__ == "__main__":
    main()
