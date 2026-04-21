# quick_convert/cli/precompute_features.py

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../configs", config_name="run/precompute_speaker_pyannote_iemocap")
def main(cfg: DictConfig) -> None:
    # Helpful for debugging composed config at runtime.
    print(OmegaConf.to_yaml(cfg, resolve=True))


    pipeline = hydra.utils.instantiate(cfg.pipeline)
    pipeline.run()


if __name__ == "__main__":
    main()
