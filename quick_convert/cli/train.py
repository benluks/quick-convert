# quick_convert/cli/train.py

from __future__ import annotations

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="run/train_bpe_tokenizer_librispeech",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    pipeline = instantiate(cfg.pipeline)
    pipeline.run()


def entrypoint() -> None:
    main()


if __name__ == "__main__":
    entrypoint()
