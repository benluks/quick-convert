# quick_convert/pipelines/training/pipeline.py

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

from quick_convert.data.base_dataset import BaseDataset
from quick_convert.training.base import BaseTrainer


class TrainingPipeline:
    def __init__(
        self,
        trainer: BaseTrainer,
        train_dataset: BaseDataset,
        exp_name: str,
        val_dataset: BaseDataset | None = None,
        test_dataset: BaseDataset | None = None,
        out_dir: PathLike | None = None,
        **kwargs,
    ) -> None:
        self.trainer = trainer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.out_dir = out_dir
        self.out_path: Path | None = None

    def prepare(self) -> Path:
        if self.out_path is not None:
            return self.out_path

        self.out_path = Path(
            self.trainer.prepare(
                train_dataset=self.train_dataset,
                out_dir=self.out_dir,
            )
        )
        return self.out_path

    def write_config(self, config: str) -> None:
        if self.out_path is None:
            raise RuntimeError("Prepare the training pipeline before writing its config.")

        self.out_path.mkdir(parents=True, exist_ok=True)
        config_path = self.out_path / "config.yaml"
        config_path.write_text(config)
        print(f"Full config written to {config_path}")

    def run(self) -> Any:
        self.prepare()
        return self.trainer.train(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
        )
