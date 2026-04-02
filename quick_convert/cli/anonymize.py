import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


@hydra.main(
    version_base=None,
    config_path="../../configs/anonymizer",
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    pipeline = instantiate(cfg.pipeline)
    pipeline.run(**cfg.get("run", {}))


if __name__ == "__main__":
    main()
