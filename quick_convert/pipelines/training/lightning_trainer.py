from typing import Iterable, Optional

import lightning as L
import torch

from quick_convert.data import BaseDataset
from quick_convert.data.index.base import Indexer
from .base_trainer import BaseTrainer


class LightningTrainer(BaseTrainer):
    def __init__(
        self,
        module: L.LightningModule,
        train_dataloader_kwargs: Optional[dict] = {},
        val_dataloader_kwargs: Optional[dict] = {},
        compile: Optional[dict] = None,
        cudnn_benchmark: Optional[bool] = None,
        ddp: Optional[dict] = None,
        precision: Optional[str] = None,
    ):

        self.module = module
        self.train_dataloader_kwargs = train_dataloader_kwargs
        self.val_dataloader_kwargs = val_dataloader_kwargs
        self.compile_cfg = compile or {"enabled": False}
        self.cudnn_benchmark = cudnn_benchmark
        self.ddp_cfg = ddp or {"enabled": False}
        self.precision = precision

    def _trainer_kwargs_with_ddp(self, kwargs: Optional[dict] = None) -> dict:
        """Merge DDP defaults into trainer kwargs when requested."""
        trainer_kwargs = dict(kwargs or {})

        if not self.ddp_cfg.get("enabled", False):
            return trainer_kwargs

        trainer_kwargs.setdefault("accelerator", self.ddp_cfg.get("accelerator", "gpu"))
        trainer_kwargs.setdefault("devices", self.ddp_cfg.get("devices", "auto"))
        trainer_kwargs.setdefault("num_nodes", self.ddp_cfg.get("num_nodes", 1))
        trainer_kwargs.setdefault("strategy", self.ddp_cfg.get("strategy", "ddp"))

        if "sync_batchnorm" in self.ddp_cfg:
            trainer_kwargs.setdefault("sync_batchnorm", self.ddp_cfg["sync_batchnorm"])

        return trainer_kwargs

    def _maybe_enable_cudnn_benchmark(self) -> None:
        """Optionally enable cuDNN benchmarking for potentially faster conv kernels."""
        if self.cudnn_benchmark is None:
            return

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = self.cudnn_benchmark

    def _maybe_compile_module(self) -> None:
        """Optionally compile selected submodules before launching fit."""
        if not self.compile_cfg.get("enabled", False):
            return

        targets = self.compile_cfg.get("targets", [])
        backend = self.compile_cfg.get("backend", "inductor")
        mode = self.compile_cfg.get("mode", "default")
        fullgraph = self.compile_cfg.get("fullgraph", False)
        dynamic = self.compile_cfg.get("dynamic", True)

        for target in targets:
            if not hasattr(self.module, target):
                continue

            submodule = getattr(self.module, target)
            if not isinstance(submodule, torch.nn.Module):
                continue

            compiled_submodule = torch.compile(
                submodule,
                backend=backend,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
            setattr(self.module, target, compiled_submodule)

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
        trainer_kwargs = self._trainer_kwargs_with_ddp(kwargs)

        # `ckpt_path` is a Trainer.fit() argument, not a Trainer() constructor
        # argument, so pull it out before building the Trainer. Pass
        # `+pipeline.train_kwargs.ckpt_path=last` (or an explicit .ckpt path) to
        # resume a timed-out / crashed run from its last checkpoint; Lightning
        # restores model + optimizer + LR scheduler + global_step + epoch.
        ckpt_path = trainer_kwargs.pop("ckpt_path", None)

        if self.precision is not None:
            trainer_kwargs.setdefault("precision", self.precision)

        self.pl_trainer = L.Trainer(default_root_dir=out_dir, **trainer_kwargs)
        self._maybe_enable_cudnn_benchmark()
        self._maybe_compile_module()

        train_loader = train_dataset.make_dataloader(**self.train_dataloader_kwargs)
        val_loader = val_dataset.make_dataloader(**self.val_dataloader_kwargs) if val_dataset else None

        return self.pl_trainer.fit(
            model=self.module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path,
        )
