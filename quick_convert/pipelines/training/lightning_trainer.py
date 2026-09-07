from pathlib import Path

import lightning as L
import torch

from quick_convert.data import BaseDataset

from .base_trainer import BaseTrainer


class LightningTrainer(BaseTrainer):
    def __init__(
        self,
        module: L.LightningModule,
        train_dataloader_kwargs: dict | None = None,
        val_dataloader_kwargs: dict | None = None,
        trainer_kwargs: dict | None = None,
        compile: dict | None = None,
        cudnn_benchmark: dict | None = None,
        ddp: dict | None = None,
        precision: dict | None = None,
    ):

        self.module = module
        self.train_dataloader_kwargs = train_dataloader_kwargs or {}
        self.val_dataloader_kwargs = val_dataloader_kwargs or {}
        self.trainer_kwargs = trainer_kwargs or {}
        self.compile_cfg = compile or {"enabled": False}
        self.cudnn_benchmark = cudnn_benchmark
        self.ddp_cfg = ddp or {"enabled": False}
        self.precision = precision

    def _trainer_kwargs_with_ddp(self, kwargs: dict | None = None) -> dict:
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

    def build(
        self,
        train_dataset,
        out_dir=None,
    ):

        # where indexing and loss building happens
        self.module.setup_training(train_dataset)
        trainer_kwargs = self._trainer_kwargs_with_ddp(self.trainer_kwargs)

        # `ckpt_path` is a Trainer.fit() argument, not a Trainer() constructor
        # argument, so pull it out before building the Trainer. Pass
        # `+pipeline.train_kwargs.ckpt_path=last` (or an explicit .ckpt path) to
        # resume a timed-out / crashed run from its last checkpoint; Lightning
        # restores model + optimizer + LR scheduler + global_step + epoch.
        self.ckpt_path = trainer_kwargs.pop("ckpt_path", None)

        if self.precision is not None:
            trainer_kwargs.setdefault("precision", self.precision)

        self.pl_trainer = L.Trainer(default_root_dir=out_dir, **trainer_kwargs)
        # self.log_dir = self.pl_trainer.log_dir
        for logger in self.pl_trainer.loggers:
            print(
                type(logger),
                logger.name,
                logger.version,
                logger.save_dir,
            )

    @property
    def log_dir(self) -> Path:
        if not self.pl_trainer.loggers:
            return Path(self.pl_trainer.default_root_dir)

        logger = self.pl_trainer.loggers[0]
        _ = logger.experiment

        save_dir = logger.save_dir if logger.save_dir is not None else self.pl_trainer.default_root_dir

        version = logger.version
        if version is None:
            raise RuntimeError(
                "A version must be associated with your logger. This usually means accessing the lazy `logger.experiment` attribute."
            )

        return Path(save_dir) / logger.name / str(version)

    def train(
        self,
        train_dataset: BaseDataset,
        val_dataset: BaseDataset | None = None,
    ):

        self._maybe_enable_cudnn_benchmark()
        self._maybe_compile_module()

        train_loader = train_dataset.make_dataloader(**self.train_dataloader_kwargs)
        val_loader = val_dataset.make_dataloader(**self.val_dataloader_kwargs) if val_dataset else None

        return self.pl_trainer.fit(
            model=self.module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=self.ckpt_path,
        )
