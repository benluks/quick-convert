from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

# from ..pipelines.evaluation import EvalPipeline


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="run/eval_asr_librispeech",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    pipeline = hydra.utils.instantiate(cfg.pipeline)
    # dataset = hydra.utils.instantiate(cfg.dataset)
    # asr = hydra.utils.instantiate(cfg.asr)
    # metric = hydra.utils.instantiate(cfg.metric)

    # pipeline = EvalPipeline(
    #     dataset=dataset,
    #     asr=asr,
    #     metric=metric,
    #     out_dir=cfg.out_dir,
    # )
    pipeline.run()


if __name__ == "__main__":
    main()
