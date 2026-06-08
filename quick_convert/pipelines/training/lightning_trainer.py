from typing import Optional

import lightning as L

from quick_convert.data import BaseDataset
from .base_trainer import BaseTrainer


class LightningTrainer(BaseTrainer):
    def __init__(
        self,
        module: L.LightningModule,
        train_dataloader_kwargs: Optional[dict] = {},
        val_dataloader_kwargs: Optional[dict] = {},
    ):

        self.module = module
        self.train_dataloader_kwargs = train_dataloader_kwargs
        self.val_dataloader_kwargs = val_dataloader_kwargs

    def train(
        self,
        train_dataset: BaseDataset,
        val_dataset: Optional[BaseDataset] = None,
        out_dir=None,
        kwargs: Optional[dict] = {},
    ):

        self.pl_trainer = L.Trainer(default_root_dir=out_dir, **kwargs)

        train_loader = train_dataset.make_dataloader(**self.train_dataloader_kwargs)
        val_loader = val_dataset.make_dataloader(**self.val_dataloader_kwargs) if val_dataset else None

        return self.pl_trainer.fit(
            model=self.module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
