# quick_convert/cli/create_manifest.py

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from .run import execute_pipeline


@hydra.main(version_base=None, config_path="../../configs", config_name="run/build_manifest_libri")
def main(cfg: DictConfig) -> None:
    execute_pipeline(cfg)


if __name__ == "__main__":
    main()
