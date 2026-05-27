# quick_convert/pipelines/training/train.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from quick_convert.pipelines.training.base_trainer import BaseTrainer


class TrainingPipeline:
    def __init__(
        self,
        trainer: BaseTrainer,
        train_dataset: Any | None = None,
        val_dataset: Any | None = None,
        test_dataset: Any | None = None,
        output_dir: str | Path | None = None,
        train_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        self.trainer = trainer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.output_dir = output_dir
        self.train_kwargs = train_kwargs

    def run(self) -> Any:
        return self.trainer.train(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            test_dataset=self.test_dataset,
            output_dir=self.output_dir,
            **self.train_kwargs,
        )
