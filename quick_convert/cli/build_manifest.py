# quick_convert/cli/create_manifest.py

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="../../configs", config_name="run/build_libri_manifest")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    pipeline = instantiate(cfg.pipeline)
    pipeline.run()


if __name__ == "__main__":
    main()
