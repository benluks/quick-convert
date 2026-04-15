from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ..pipelines.asv.utils import find_embedding_model_ckpt
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

    return tuple(
        [
            str(p) if p.is_file() else None
            for p in [train_csv, enrol_csv, test_csv, trials_txt]
        ]
    )


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

    if (not train_csv) and (cfg.asv.overrides.score_norm):
        raise ValueError(
            "No train_csv found. Please make sure you provide one or set `score_norm` to None"
        )

    if not all([train_csv, enrol_csv, test_csv, trials_txt]) or cfg.asv.overwrite_csv:
        enrol_csv, test_csv, trials_txt = prepare_asv_eval_data(
            cfg.asv.mode, **cfg.prep
        )

    emb_ckpt = find_embedding_model_ckpt(
        cfg.asv.overrides.save_folder,
    )

    overrides = {
        **cfg.asv.overrides,
        "train_data": train_csv,
        "enrol_data": enrol_csv,
        "test_data": test_csv,
        "verification_file": trials_txt,
        "data_folder": "/unused/by_custom_prep",
        "skip_prep": True,
        "pretrain_path": str(emb_ckpt.parent)
    }

    eval_asv(
        hparams_file=cfg.asv.eval_hparams_file,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
