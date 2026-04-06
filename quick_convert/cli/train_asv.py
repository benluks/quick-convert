from __future__ import annotations

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ..pipelines.asv.prepare_dataset import prepare_asv_csvs_from_dataset
from ..pipelines.asv.train import train_asv


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="run/train_asv_clac",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    dataset = instantiate(cfg.dataset)

    train_csv, dev_csv, n_speakers = prepare_asv_csvs_from_dataset(
        dataset=dataset,
        save_folder=cfg.asv.prep.save_folder,
        train_fraction=cfg.asv.prep.train_fraction,
        seed=cfg.asv.prep.seed,
        randomize_within_split=cfg.asv.prep.randomize_within_split,
    )

    overrides = {
    "output_folder": cfg.asv.overrides.output_folder,
    "save_folder": cfg.asv.overrides.save_folder,
    "data_folder": cfg.asv.overrides.data_folder,
    "train_annotation": train_csv,
    "valid_annotation": dev_csv,
    "out_n_neurons": n_speakers,
    "skip_prep": True,
}

    train_asv(
        hparams_file=cfg.asv.hparams_file,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
