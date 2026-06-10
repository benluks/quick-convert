from typing import Iterable, Optional

import lightning as L

from quick_convert.data import BaseDataset
from quick_convert.data.index.base import Indexer
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
        for index in self.module.indexers.values():
            index.fit(train_dataset.rows)

        # build the losses that are dependent on some other indexed value, and therefore 
        # couldn't be passed in the hydra config. For example, if the speaker identification
        # index is only build in the line above, then we wouldn't know the number of output classes
        # needed until now. The output projection layer is built in the loss (e.g. `AAMSoftmaxLoss()`)
        for module in self.module.modules():
            if module is self:
                continue
            if hasattr(module, "build_loss"):
                module.build_loss(self.module.indexers)

        self.pl_trainer = L.Trainer(default_root_dir=out_dir, **kwargs)

        train_loader = train_dataset.make_dataloader(**self.train_dataloader_kwargs)
        val_loader = val_dataset.make_dataloader(**self.val_dataloader_kwargs) if val_dataset else None

        return self.pl_trainer.fit(
            model=self.module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
