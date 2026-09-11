import hydra
from omegaconf import DictConfig

from .run import execute_pipeline


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="run/anonymize_asrbn_clac",
)
def main(cfg: DictConfig):
    return execute_pipeline(cfg)


if __name__ == "__main__":
    main()
