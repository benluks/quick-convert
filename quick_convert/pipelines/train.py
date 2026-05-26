# quick_convert/pipelines/training/module.py

from __future__ import annotations

from typing import Any, Optional

import lightning as L
from quick_convert.data.base_dataset import BaseDataset


class ModuleTrainingPipeline:
    """
    Generic Lightning module training pipeline.

    Expects datasets that expose `.make_dataloader(...)`.
    """

    def __init__(
        self,
        module: L.LightningModule,
        train_dataset: BaseDataset,
        val_dataset: Optional[BaseDataset] = None,
        test_dataset: Optional[BaseDataset] = None,
        train_dataloader_kwargs: Optional[dict[str, Any]] = {},
        val_dataloader_kwargs: Optional[dict[str, Any]] = {},
        test_dataloader_kwargs: Optional[dict[str, Any]] = {},
        trainer_kwargs: Optional[dict[str, Any]] = {},
        run_test: bool = False,
    ) -> None:
        self.module = module

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.train_dataloader_kwargs = train_dataloader_kwargs or {}
        self.val_dataloader_kwargs = val_dataloader_kwargs or {}
        self.test_dataloader_kwargs = test_dataloader_kwargs or {}

        self.trainer = L.Trainer(**trainer_kwargs)
        self.run_test = run_test

    def _make_dataloader(self, dataset: Any, kwargs: dict[str, Any]):
        if dataset is None:
            return None

        if not hasattr(dataset, "make_dataloader"):
            raise TypeError(f"{type(dataset).__name__} does not expose `make_dataloader(...)`.")

        return dataset.make_dataloader(**kwargs)

    def run(self) -> None:
        train_loader = self._make_dataloader(
            self.train_dataset,
            self.train_dataloader_kwargs,
        )

        val_loader = self._make_dataloader(
            self.val_dataset,
            self.val_dataloader_kwargs,
        )

        self.trainer.fit(
            model=self.trainer,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        if self.run_test:
            test_loader = self._make_dataloader(
                self.test_dataset,
                self.test_dataloader_kwargs,
            )

            if test_loader is None:
                raise ValueError("run_test=True, but no test_dataset was provided.")

            self.trainer.test(
                model=self.trainer,
                dataloaders=test_loader,
            )
