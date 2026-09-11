# quick_convert/cli/train.py

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from .run import execute_pipeline


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="run/train_bpe_tokenizer_librispeech",
)
def main(cfg: DictConfig) -> None:
    execute_pipeline(cfg)


def entrypoint() -> None:
    main()


if __name__ == "__main__":
    entrypoint()
