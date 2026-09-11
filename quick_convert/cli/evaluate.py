from __future__ import annotations

import hydra
from omegaconf import DictConfig

from .run import execute_pipeline


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="run/eval_asr_librispeech",
)
def main(cfg: DictConfig) -> None:
    execute_pipeline(cfg)


if __name__ == "__main__":
    main()
