from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="run/eval_asr_librispeech",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    pipeline = hydra.utils.instantiate(cfg.pipeline)
    pipeline.run()


if __name__ == "__main__":
    main()
