# quick_convert/pipelines/training/train.py

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Optional

from quick_convert.data.base_dataset import BaseDataset
from quick_convert.pipelines.training.base_trainer import BaseTrainer


class TrainingPipeline:
    def __init__(
        self,
        trainer: BaseTrainer,
        train_dataset: BaseDataset,
        val_dataset: Optional[BaseDataset] = None,
        test_dataset: Optional[BaseDataset] | None = None,
        out_dir: PathLike = None,
        train_kwargs: Optional[dict] = {},
        **kwargs,
    ) -> None:
        self.trainer = trainer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.out_dir = out_dir
        self.train_kwargs = train_kwargs

    def run(self) -> Any:
        return self.trainer.train(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            out_dir=self.out_dir,
            kwargs=self.train_kwargs,
        )
