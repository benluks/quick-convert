from __future__ import annotations
import csv
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ..pipelines.asv.prepare_dataset import prepare_asv_csvs_from_dataset
from ..pipelines.asv.train import train_asv


def count_unique_speakers(csv_path: str | Path) -> int:
    speakers = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            speakers.add(row["spk_id"])
    return len(speakers)


def resolve_prepared_data_path(
    prepared_data_path: str | Path,
) -> tuple[Path, Path] | None:
    prepared_data_path = Path(prepared_data_path)
    if not prepared_data_path.is_dir():
        return None

    train_csv = prepared_data_path / "train.csv"
    dev_csv = prepared_data_path / "dev.csv"

    if train_csv.is_file() and dev_csv.is_file():
        return train_csv, dev_csv

    return None, None


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="run/train_asv_clac",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # check if train and dev csvs exist in prepared data path
    train_csv, dev_csv = resolve_prepared_data_path(cfg.asv.prepared_data_path)
    if train_csv and dev_csv:
        n_speakers = count_unique_speakers(train_csv)
    else:
        dataset = instantiate(cfg.dataset)

        train_csv, dev_csv, n_speakers = prepare_asv_csvs_from_dataset(
            dataset=dataset,
            save_folder=cfg.asv.prep.save_folder,
            train_fraction=cfg.asv.prep.train_fraction,
            seed=cfg.asv.prep.seed,
            randomize_within_split=cfg.asv.prep.randomize_within_split,
        )

    overrides = {
        **cfg.asv.overrides,
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
