from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ..pipelines.asv.prepare_dataset import (
    prepare_asv_eval_data,
)
from ..pipelines.asv.eval import eval_asv


def resolve_prepared_verification_path(
    prepared_data_path: str | Path | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    if not prepared_data_path:
        return None, None, None, None

    prepared_data_path = Path(prepared_data_path)
    if not prepared_data_path.is_dir():
        return None, None, None, None

    train_csv = prepared_data_path / "train.csv"
    enrol_csv = prepared_data_path / "enrol.csv"
    test_csv = prepared_data_path / "test.csv"
    trials_txt = prepared_data_path / "trials.txt"

    if all(p.is_file() for p in [train_csv, enrol_csv, test_csv, trials_txt]):
        return (
            str(train_csv),
            str(enrol_csv),
            str(test_csv),
            str(trials_txt),
        )

    return None, None, None, None


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="run/eval_asv_clac",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    train_csv, enrol_csv, test_csv, trials_txt = resolve_prepared_verification_path(
        cfg.asv.prepared_data_path
    )

    if not all([train_csv, enrol_csv, test_csv, trials_txt]) or cfg.asv.overwrite_csv:
        enrol_csv, test_csv, trials_txt = prepare_asv_eval_data(
            cfg.asv.mode, **cfg.prep
        )

    overrides = {
        **cfg.asv.overrides,
        "train_data": train_csv,
        "enrol_data": enrol_csv,
        "test_data": test_csv,
        "verification_file": trials_txt,
        "data_folder": "/unused/by_custom_prep",
        "skip_prep": True,
    }

    eval_asv(
        hparams_file=cfg.asv.verification_hparams_file,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
