import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="run/anonymization_asrbn_clac",
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    anonymizer = instantiate(cfg.anonymizer)
    dataset = instantiate(cfg.dataset)
    pipeline = instantiate(
        cfg.pipeline,
        anonymizer=anonymizer,
        dataset=dataset,
    )

    pipeline.run(**cfg.get("run", {}))


if __name__ == "__main__":
    main()
