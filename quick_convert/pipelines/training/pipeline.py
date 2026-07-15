# quick_convert/pipelines/training/pipeline.py

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
        **kwargs,
    ) -> None:
        self.trainer = trainer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.out_dir = out_dir

        # separate build and train so that trainer.save_config can be called in between, because I want all pipelines to have that
        # options, but the lightningmodule needs to have been built to expose the log dir to which the yaml config should be saved
        # the `build` function on a trainer make the output dir accessible, because `write_config` is called on the pipeline level.
        self.trainer.build(
            train_dataset=self.train_dataset,
            # kwargs=self.train_kwargs,
            out_dir=self.out_dir,
        )
        self.out_path = Path(self.trainer.log_dir)

    # TODO: abstract this abstract pipeline class
    def write_config(self, config):

        self.out_path.mkdir(parents=True, exist_ok=True)
        config_path = self.out_path / "config.yaml"
        config_path.write_text(config)
        print(f"Full config written to {config_path}")

    def run(self) -> Any:
        return self.trainer.train(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            # out_dir=self.out_dir,
            # kwargs=self.train_kwargs,
        )
